"""
@motjuste
Created: 01-02-2017

Helpers for working with Fisher dataset
"""
from __future__ import print_function, division
from collections import namedtuple, Iterable
import os
from csv import reader
import numpy as np
import warnings

import rennet.utils.label_utils as lu
from rennet.utils.np_utils import group_by_values

# TODO: Extract gender?
# Will require reading the extra database files or something
FisherSpeaker = namedtuple('FisherSpeaker', ['speakerid'])

# TODO: Extract confidence?
# Will require reading the extra database files or something
FisherTranscription = namedtuple('FisherTranscription',
                                 ['speakerid', 'content'])


class FisherAnnotations(lu.SequenceLabels):
    """
    TODO: [ ] Add proper docs

    NOTE: This is almost identical to ka3.Annotations, but copied here, cuz
    - Fisher is the main dataset for me.
    - ka3 module was not designed to be sub-classed.
    - Don't want headache of maintaining compatibility where I don't have to
    """

    def __init__(self, filepath, speakers, *args, **kwargs):
        self.sourcefile = filepath
        self.speakers = sorted(speakers, key=lambda s: s.speakerid)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_file(cls, filepath):
        afp = os.path.abspath(filepath)

        se = []
        sid = set()
        trans = []

        with open(afp, 'r') as f:
            rdr = reader(f, delimiter=':')

            for row in rdr:
                if len(row) == 0 or row[0][0] == '#':
                    # ignore empty lines or comments
                    continue
                else:
                    s, e, spk = row[0].split(' ')
                    spk = spk.strip()
                    content = row[1].strip()

                    sid.add(spk)
                    se.append((float(s), float(e)))
                    trans.append(FisherTranscription(spk, content))

        speakers = [FisherSpeaker(id) for id in sid]

        return cls(afp, speakers, se, trans, samplerate=1)

    def idx_for_speaker(self, speaker):
        speakerid = speaker.speakerid
        for i, l in enumerate(self.labels):
            if l.speakerid == speakerid:
                yield i

    def __str__(self):
        s = "Source filepath: {}".format(self.sourcefile)
        s += "\nSpeakers: {}\n".format(len(self.speakers))
        s += "\n".join(str(s) for s in self.speakers)
        s += "\n" + super().__str__()
        return s


class FisherActiveSpeakers(FisherAnnotations):
    def __init__(self, filepath, speakers, *args, **kwargs):
        super().__init__(filepath, speakers, *args, **kwargs)
        self.labels = np.array(
            self.labels)  # SequenceLabels makes it into a list

    @classmethod
    def from_annotations(cls, ann, samplerate=100,
                         warn=True):  # min time resolution 1ms, mostly
        """
        TODO: [ ] Better handling of warnings?
            The user should be aware that there is a problem,
            and some implicit decisions were made
            Hence `warn = True` by default
        """
        with ann.samplerate_as(samplerate):
            _se = ann.starts_ends
            se = np.round(_se).astype(np.int)

        if warn:
            try:
                np.testing.assert_almost_equal(se, _se)
            except AssertionError:
                _w = "Sample rate {} does not evenly divide all the starts and ends for file {}".format(
                    samplerate, ann.sourcefile)
                warnings.warn(_w)

        # make contigious array of shape (total_duration, n_speakers)
        active_speakers = np.zeros(
            shape=(se[:, 1].max(), len(ann.speakers)), dtype=np.int)

        for s, speaker in enumerate(ann.speakers):
            for i in ann.idx_for_speaker(speaker):
                start, end = se[i]

                # NOTE: not setting to 1 straighaway to catch duplicates
                active_speakers[start:end, s] += 1

        if active_speakers.max() > 1:
            if warn:
                _w = "Some speakers may have duplicate annotations for file {}.\n!!! IGNORED !!!".format(
                    ann.sourcefile)
                warnings.warn(_w)

            active_speakers[active_speakers > 1] = 1

        starts_ends, active_speakers = group_by_values(active_speakers)

        return cls(ann.sourcefile,
                   ann.speakers,
                   starts_ends,
                   active_speakers,
                   samplerate=samplerate)

    @classmethod
    def from_file(cls, filepath):
        ann = super().from_file(filepath)
        return cls.from_annotations(
            ann, samplerate=100)  # min time resolution 1ms, mostly

    def labels_at(self, ends, samplerate=None):
        """
        NOTE: Here and not in the base class cuz
            The segments are contigious and guaranteed non-overlapping
        """
        if not isinstance(ends, Iterable):
            ends = [ends]

        ends = np.array(ends)

        # NOTE: To make sure that we are working with the correct samplerate

        # the samplerate could be set temporarily in a different context
        in_diffcontext = self._samplerate != self._orig_samplerate

        if samplerate is None or samplerate == self._orig_samplerate:
            # Assume that the samplerate of given ends == self.samplerate
            # irrespective of being in a different context
            se = self.starts_ends
        elif in_diffcontext:
            # temporarily change to required samplerate context
            # then change back to the previous context
            # TODO: [ ] test this, properly, probably
            ctx_samplerate = self._samplerate
            with self.samplerate_as(samplerate):
                se = self.starts_ends

            self._samplerate = ctx_samplerate
        else:
            # temporarily change to required samplerate context
            with self.samplerate_as(samplerate):
                se = self.starts_ends

        endings = se[:, 1]
        maxend = endings.max()
        minstart = se[:, 0].min()
        endswithin = (ends <= maxend) & (ends >= minstart)

        # find labels for valid ends that are smaller than or equal to endings
        label_idx = np.searchsorted(endings, ends[endswithin], side='left')

        # NOTE: Can't resolve for ends that are not indside,
        #   will assume to spit out zeros
        labels = np.zeros((len(ends), *self.labels.shape[1:]), dtype=np.int)
        labels[endswithin] = self.labels[label_idx]

        return labels
