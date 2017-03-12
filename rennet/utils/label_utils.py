"""
@motjuste
Created: 26-08-2016

Utilities for working with labels
"""
from __future__ import print_function, division
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
    TODO: [ ] Export to ELAN
    """

    __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate')

    def __init__(self, starts_ends, labels, samplerate=1):
        assert all(isinstance(x, Iterable)
                   for x in [starts_ends, labels]), "starts_ends and labels" + \
                                                    " should be iterable"
        assert len(starts_ends) == len(labels), "starts_ends and labels" + \
                                                " mismatch in length "

        starts_ends = np.array(starts_ends)
        assert np.all(starts_ends[:, 1] - starts_ends[:, 0] >
                      0.), "(ends - starts) should be > 0 for all pairs"

        sidx = np.argsort(starts_ends[:, 0])  # save sorted by starts
        self._starts_ends = starts_ends[sidx]

        if isinstance(labels, np.ndarray):
            self.labels = labels[sidx]
        else:
            self.labels = [labels[i] for i in sidx]

        self._orig_samplerate = samplerate
        self._samplerate = samplerate

    @property
    def samplerate(self):
        # If you want to set a different samplerate, use samplerate_as
        # If you have made a mistake, create a new instance
        return self._samplerate  # always the current samplerate

    @property
    def orig_samplerate(self):
        # Changing the original samplerate may lead to wrong results
        # hence provided as a property to only read
        # No one's stopping them. Hoping the user is an adult
        return self._orig_samplerate

    @property
    def starts_ends(self):
        return self._starts_ends * (self.samplerate / self.orig_samplerate)

    @property
    def starts(self):
        return self.starts_ends[:, 0]

    @property
    def ends(self):
        return self.starts_ends[:, 1]

    @contextmanager
    def samplerate_as(self, new_sr):
        """ Temporarily change to a different samplerate to calculate values

        To be used with a `with` clause.

        The starts and ends will calculated on the contextually most up-to-date
        samplerate.
        """
        old_sr = self.samplerate
        self._samplerate = new_sr
        try:
            yield
        finally:
            self._samplerate = old_sr

    def _starts_ends_for_samplerate(self, samplerate):
        # Available to children classes and not expected to change in context
        if samplerate == self.samplerate:
            return self.starts_ends
        else:
            # Change the context to the new samplerate
            # even if the self.samplerate was set to a different contextual one
            # the starts_ends will be calculated for given samplerate
            # and the contextual samplerate will be returned
            with self.samplerate_as(samplerate):
                return self.starts_ends

    def labels_at(self, ends, samplerate=None, default_label=()):
        if not isinstance(ends, Iterable):
            ends = [ends]

        ends = np.array(ends)

        # make sure we are working with the correct samplerate for starts_ends
        if samplerate is None:
            # Assume the user is expecting the current samplerate
            # self.samplerate always has the most up to date samplerate
            se = self.starts_ends
        else:
            se = self._starts_ends_for_samplerate(samplerate)

        se = np.round(se, 10)  # To avoid issues with floating points
        # Yes, it looks arbitrary
        # There will be problems later in comparing floating point numbers
        # esp when testing for equality to the ends
        # the root of the cause is when the provided samplerate is higher than
        # the orig_samplerate, esp when the ratio is irrational.
        # A max disparity of 16000:1 gave correct results
        # limited to the tests of course. Please look at the corresponding tests
        #
        # FIXME: It should not be the case anyway. All the trouble to support
        #       arbitrary self.orig_samplerate and samplerate
        #

        res = []
        for e in ends:
            l_idx = np.where((se[:, 0] < e) & (se[:, 1] >= e))[0]
            if len(l_idx) == 0:  # end not within segments
                res.append(default_label)
            else:
                res.append(tuple(self.labels[i] for i in l_idx))

        return res

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


class ContiguousSequenceLabels(SequenceLabels):
    """ Special SequenceLabels with contiguous labels

    There is a label for each sample between min(starts) and max(ends)

    """

    # PARENT'S SLOTS
    # __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate')
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(ContiguousSequenceLabels, self).__init__(*args, **kwargs)
        # the starts_ends were sorted in __init__ on starts
        assert np.all(np.diff(self.starts) >
                      0.), "There are multiple segments with the same starts"
        assert np.all(
            self.starts_ends[1:, 0] == self.starts_ends[:-1, 1]
        ), "All ends should be the starts of the next segment, except the last"

        # convert labels to np.array
        # this is a bit controversial, since the conversion may lead
        # to some unexpected results. Unexpected for a n00b like me at least.
        # eg. list of namedtuple get converted to array of tuples.
        # user small, __slot__ classes in that case
        self.labels = np.array(self.labels)
        # IDEA: store only the unique values? min_start and ends?
        # May be pointless here in python

    def labels_at(self, ends, samplerate=None, default_label=None):
        if not isinstance(ends, Iterable):
            ends = [ends]

        ends = np.array(ends)

        # make sure we are working with the correct samplerate for starts_ends
        if samplerate is None:
            # Assume the user is expecting the current samplerate
            # self.samplerate always has the most up to date samplerate
            se = self.starts_ends
        else:
            se = self._starts_ends_for_samplerate(samplerate)

        se = np.round(se, 10)  # To avoid issues with floating points
        # Yes, it looks arbitrary. Check SequenceLabels.labels_at(...)
        #
        # FIXME: It should not be the case anyway. All the trouble to support
        #       arbitrary self.orig_samplerate and samplerate
        #

        # all ends are within the segments
        endings = se[:, 1]
        maxend = endings.max()
        minstart = endings.min()
        endswithin = (ends > minstart) & (ends <= maxend)

        allwithin = len(endswithin) == len(ends)  # no default label required

        if allwithin:
            # find indices of the labels for each ends
            label_idx = np.searchsorted(endings, se[:, 1], side='left')

            # pick the labels at those indices, and return
            return self.labels[label_idx]

        else:
            raise NotImplementedError()
