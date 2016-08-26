"""
@motjuste
Created: 26-08-2016

Utilities for working with labels
"""
import numpy as np
from collections import Iterable


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

    def __init__(self, starts_ends, labels, samplerate=1, ids=None):
        assert all(isinstance(x, Iterable)
                   for x in [starts_ends, labels
                             ]), "starts, ends, and labels should be iterable"

        self._starts_ends = np.array(starts_ends, np.int)
        self.labels = labels
        self.samplerate = samplerate
        self.ids = np.arange(len(labels)) if ids is None else ids

    def get_starts(self, samplerate=None):
        samplerate = self.samplerate if samplerate is None else samplerate
        return self._starts_ends[:, 0] / samplerate

    starts = property(get_starts)

    def get_ends(self, samplerate=None):
        samplerate = self.samplerate if samplerate is None else samplerate
        return self.ends_ends[:, 1] / samplerate

    ends = property(get_ends)

    def get_starts_ends(self, samplerate=None):
        samplerate = self.samplerate if samplerate is None else samplerate
        return self.starts_ends_starts_ends / samplerate

    starts_ends = property(get_starts_ends)
