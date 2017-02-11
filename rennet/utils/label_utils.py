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

    By default the samplerate is 1, but a default one can be set at the time
    of instantiating. The samplerate should reflect the one used in calculating
    the starts_ends.

    The starts, ends and starts_ends can be retrieved at a different
    samplerate by using the `with SequenceLabel.samplerate_as(new_samplerate)`.
    While in the scope of the `with` context will act as if the samplerate is
    set to `new_samplerate`, except `SequenceLabel.samplerate`, which will
    always return the original samplerate.

    Supports normal slicing as in numpy, but the returned value will be another
    instance of the SequenceLabels class.

    This class is not a monolith, but should be able to work with normal
    numpy tricks. Plus, you should extend it for specific data, etc.

    Plus, there is nice printing.

    TODO: [A] Check if something can be done about plotting it nicely too!
    """

    def __init__(self, starts_ends, labels, samplerate=1):
        assert all(isinstance(x, Iterable)
                   for x in [starts_ends, labels]), "starts_ends and labels" + \
                                                    " should be iterable"
        assert len(starts_ends) == len(labels), "starts_ends and labels" + \
                                                " mismatch in length "

        starts_ends = np.array(starts_ends)
        assert np.all(starts_ends[:, 1] - starts_ends[:, 0] >
                      0.), "(ends - starts) should be > 0 for all pairs"

        sidx = np.argsort(starts_ends[:, 0])

        self._starts_ends = starts_ends[sidx]  # save sorted by starts
        self.labels = [labels[i] for i in sidx]
        self._orig_samplerate = samplerate
        self._samplerate = self._orig_samplerate

    @property
    def samplerate(self):
        return self._orig_samplerate  # always the original samplerate

    @property
    def starts_ends(self):
        return self._starts_ends * (self._samplerate / self._orig_samplerate)

    @property
    def starts(self):
        return self.starts_ends[:, 0]

    @property
    def ends(self):
        return self.starts_ends[:, 1]

    @contextmanager
    def samplerate_as(self, new_sr):
        self._samplerate = new_sr
        try:
            yield
        finally:
            self._samplerate = self._orig_samplerate

    def __len__(self):
        return len(self.starts_ends)

    def __getitem__(self, idx):
        se = self._starts_ends[idx]
        l = self.labels[idx]

        if isinstance(idx, int):  # case with only one segment
            se = np.expand_dims(se, axis=0)  # shape (2,) to shape (1, 2)
            l = [l]

        sr = self.samplerate

        return SequenceLabels(se, l, sr)

    def __str__(self):
        s = self.__class__.__name__ + " with sample rate: " + str(
            self.samplerate)
        s += "\n"
        s += "{:8} - {:8} : {}\n".format("Start", "End", "Label")
        s += "\n".join("{:<8.4f} - {:<8.4f} : {}".format(s, e, str(l))
                       for (s, e), l in zip(self.starts_ends, self.labels))

        return s
