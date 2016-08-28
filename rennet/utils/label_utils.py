"""
@motjuste
Created: 26-08-2016

Utilities for working with labels
"""
import numpy as np
from collections import Iterable
from contextlib import contextmanager


class SequenceLabels(object):
    """ Base class for working with contiguous labels for sequences

    Supports normal slicing as in numpy, but the returned value will be another
    instance of the SequenceLabels class.

    NOTE: You can get the properties `starts` and `ends` at a different
    sample-rate by passing a sample-rate other than the default value of 1 to
    `get_starts`, or setting the property `samplerate` to change the default.

    This class is not a monolith, but should be able to work with normal
    numpy tricks. Plus, you should extend it for specific data, etc.

    Plus, there is nice printing.

    TODO: [P] Check if something can be done about plotting it nicely too!
    """

    def __init__(self, starts_ends, labels, npersec=1, ids=None):
        assert all(isinstance(x, Iterable)
                   for x in [starts_ends, labels
                             ]), "starts, ends, and labels should be iterable"

        self._starts_ends = np.array(starts_ends)
        self.labels = labels
        self._samplerate = npersec
        self._tempsr = self._samplerate
        self.ids = np.arange(len(labels)) if ids is None else ids

    @property
    def npersec(self):
        return self._samplerate

    @property
    def starts_ends(self):
        return self._starts_ends * (self._tempsr / self._samplerate)

    @property
    def starts(self):
        return self.starts_ends[:, 0]

    @property
    def ends(self):
        return self.starts_ends[:, 1]

    @contextmanager
    def npersec_as(self, new_sr):
        self._tempsr = new_sr
        try:
            yield
        finally:
            self._tempsr = self._samplerate
