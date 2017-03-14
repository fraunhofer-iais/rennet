"""
@motjuste
Created: 01-02-2017

Helpers for working with Fisher dataset
"""
from __future__ import print_function, division
import os
from csv import reader
import numpy as np
from collections import namedtuple
import warnings

import rennet.utils.label_utils as lu
from rennet.utils.np_utils import group_by_values


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
    __slots__ = ('sourcefile', 'calldata')

    FisherTranscription = namedtuple('FisherTranscription',
                                     ['speakerchannel', 'content'])


    def __init__(self, filepath, calldata, *args, **kwargs):
        self.sourcefile = filepath
        if calldata is None:
            self.calldata = calldata
        else:
            raise NotImplementedError("Reading FisherCalldata not implemented")

        super(FisherAnnotations, self).__init__(*args, **kwargs)

    @property
    def callid(self):
        # filenames are fe_03_CALLID.*
        return os.path.basename(self.sourcefile).split('_')[-1].split('.')[0]

    @classmethod
    def from_file(cls, filepath, allcalldata=None):
        afp = os.path.abspath(filepath)

        se = []
        trans = []

        with open(afp, 'r') as f:
            rdr = reader(f, delimiter=':')

            for row in rdr:
                if len(row) == 0 or row[0][0] == '#':
                    # ignore empty lines or comments
                    continue
                else:
                    s, e, spk = row[0].split(' ')
                    se.append((float(s), float(e)))

                    spk = spk.strip()
                    content = row[1].strip()
                    if spk.upper() == 'A':
                        trans.append(cls.FisherTranscription(0, content))
                    elif spk.upper() == 'B':
                        trans.append(cls.FisherTranscription(1, content))
                    else:
                        raise ValueError(
                            "Speaker channel other than A and B ({}) in file\n{}".
                            format(spk, filepath))

        if allcalldata is None:
            calldata = None
        else:
            raise NotImplementedError("Reading FisherCalldata not implemented")

        return cls(afp, calldata, se, trans, samplerate=1)

    def __str__(self):
        s = "Source filepath: {}".format(self.sourcefile)
        s += "\nCalldata: {}".format(self.calldata)
        s += "\n" + super(FisherAnnotations, self).__str__()
        return s


class FisherActiveSpeakers(lu.ContiguousSequenceLabels):
    # PARENT'S SLOTS
    # __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate')
    __slots__ = ('sourcefile', 'calldata')

    def __init__(self, filepath, calldata, *args, **kwargs):
        self.sourcefile = filepath
        self.calldata = calldata

        super(FisherActiveSpeakers, self).__init__(*args, **kwargs)

        # SequenceLabels makes labels into a list
        self.labels = np.array(self.labels)

    @property
    def callid(self):
        # filenames are fe_03_CALLID.*
        return os.path.basename(self.sourcefile).split('_')[-1].split('.')[0]

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
        # NOTE: n_speakers is 2 for all Fisher data
        n_speakers = 2
        active_speakers = np.zeros(
            shape=(se[:, 1].max(), n_speakers), dtype=np.int)

        for (start, end), l in zip(se, ann.labels):
            # NOTE: not setting to 1 straightaway to catch duplicates
            active_speakers[start:end, l.speakerchannel] += 1

        if active_speakers.max() > 1:
            if warn:
                _w = "Some speakers may have duplicate annotations for file:\n{}.\n!!! IGNORED !!!".format(
                    ann.sourcefile)
                warnings.warn(_w)

            active_speakers[active_speakers > 1] = 1

        starts_ends, active_speakers = group_by_values(active_speakers)

        return cls(ann.sourcefile,
                   ann.calldata,
                   starts_ends,
                   active_speakers,
                   samplerate=samplerate)

    @classmethod
    def from_file(cls, filepath, samplerate=100, allcalldata=None, warn=True):
        ann = FisherAnnotations.from_file(filepath, allcalldata)
        # min time resolution 1ms, mostly
        return cls.from_annotations(ann, samplerate=samplerate, warn=warn)

    def __str__(self):
        s = "Source filepath: {}".format(self.sourcefile)
        s += "\nCalldata: {}".format(self.calldata)
        s += "\n" + super(FisherActiveSpeakers, self).__str__()
        return s
