"""
@motjuste
Created: 01-02-2017

Helpers for working with Fisher dataset
"""
from __future__ import print_function, division
from collections import namedtuple
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

    # PARENT'S SLOTS
    # __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate')
    __slots__ = ('sourcefile', 'speakers')

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


class FisherActiveSpeakers(lu.ContiguousSequenceLabels):
    # PARENT'S SLOTS
    # __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate')
    __slots__ = ('sourcefile', 'speakers')

    def __init__(self, filepath, speakers, *args, **kwargs):
        self.sourcefile = filepath
        self.speakers = sorted(speakers, key=lambda s: s.speakerid)

        super().__init__(*args, **kwargs)

        # SequenceLabels makes labels into a list
        self.labels = np.array(self.labels)

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
                _w = "Sample rate {} does not evenly divide all the starts and ends for file:\n{}".format(
                    samplerate, ann.sourcefile)
                warnings.warn(_w)

        # make contigious array of shape (total_duration, n_speakers)
        active_speakers = np.zeros(
            shape=(se[:, 1].max(), len(ann.speakers)), dtype=np.int)

        for s, speaker in enumerate(ann.speakers):
            for i in ann.idx_for_speaker(speaker):
                start, end = se[i]

                # NOTE: not setting to 1 straightaway to catch duplicates
                active_speakers[start:end, s] += 1

        if active_speakers.max() > 1:
            if warn:
                _w = "Some speakers may have duplicate annotations for file:\n{}.\n!!! IGNORED !!!".format(
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
    def from_file(cls, filepath, samplerate=100, warn=True):
        ann = FisherAnnotations.from_file(filepath)

        # min time resolution 1ms, mostly
        return cls.from_annotations(ann, samplerate=samplerate, warn=warn)

    def __str__(self):
        s = "Source filepath: {}".format(self.sourcefile)
        s += "\nSpeakers: {}\n".format(len(self.speakers))
        s += "\n".join(str(s) for s in self.speakers)
        s += "\n" + super().__str__()
        return s
