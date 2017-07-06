"""
@motjuste
Created: 28-08-2016

Helpers for working with TIMIT dataset
"""
from os.path import abspath
from csv import reader
import numpy as np

import rennet.utils.label_utils as lu
from rennet.utils.np_utils import group_by_values


class Annotations(lu.SequenceLabels):
    # PARENT'S SLOTS
    # __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate')
    __slots__ = ('sourcefile', )

    def __init__(self, filepath, *args, **kwargs):
        self.sourcefile = filepath
        super(Annotations, self).__init__(*args, **kwargs)

    @classmethod
    def from_file(cls, filepath, delimiter=' ', samplerate=16000):

        se = []
        l = []

        absfp = abspath(filepath)
        with open(absfp) as f:
            rdr = reader(f, delimiter=delimiter)

            for row in rdr:
                se.append((int(row[0]), int(row[1])))
                l.append(" ".join(row[2:]))

        return cls(absfp, se, l, samplerate=samplerate)

    def overlay(self, other, samplerate=None):
        """ Overlay the TIMIT annotation with another one
        NOTE: The other annotation will be clipped if it is longer than the current one
        """

        assert isinstance(
            other, lu.SequenceLabels
        ), "The other label should be a TIMIT single speaker label"

        if samplerate is None:
            samplerate = self.samplerate

        with self.samplerate_as(samplerate):
            se = np.round(self.starts_ends).astype(np.int)

        total_duration = se[:, 1].max()

        active_speakers = np.zeros(shape=(total_duration, 2), dtype=np.int)

        for s, e in se:
            active_speakers[s:e, 0] = 1

        with other.samplerate_as(samplerate):
            se_other = np.round(other.starts_ends).astype(np.int)

        for s, e in se_other:
            if s - 1 < total_duration:
                if e < total_duration:
                    active_speakers[s:e, 1] = 1
                else:
                    active_speakers[s:, 1] = 1
            else:
                continue

        starts_ends, active_speakers = group_by_values(active_speakers)

        return self.__class__(
            self.sourcefile,
            starts_ends,
            active_speakers,
            samplerate=samplerate)

    def __getitem__(self, idx):
        args = super(Annotations, self).__getitem__(idx)
        if self.__class__ is Annotations:
            return self.__class__(self.sourcefile, *args)
        else:
            return args

    def __str__(self):
        s = "Source filepath:\n{}\n".format(self.sourcefile)
        s += "\n" + super(Annotations, self).__str__()
        return s
