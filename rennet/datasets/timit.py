"""
@motjuste
Created: 28-08-2016

Helpers for working with TIMIT dataset
"""
from os.path import abspath
from csv import reader

import rennet.utils.label_utils as lu


class TIMITSequenceLabels(lu.SequenceLabels):
    def __init__(self, filepath, *args, **kwargs):
        self.sourcefile = filepath
        super().__init__(*args, **kwargs)

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
