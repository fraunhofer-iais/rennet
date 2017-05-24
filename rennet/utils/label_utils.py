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
        """Initialize a SequenceLabels instance with starts_ends and labels"""
        # TODO: [ ] Add dox, at least the params and attributes

        if any(not isinstance(x, Iterable) for x in [starts_ends, labels]):
            raise TypeError("starts_ends and labels should be Iterable")
        if len(starts_ends) != len(labels):
            raise AssertionError("starts_ends and labels mismatch in length")

        starts_ends = np.array(starts_ends)
        if len(starts_ends.shape) != 2 or starts_ends.shape[-1] != 2:
            raise AssertionError(
                "starts_ends doesn't look like a list of pairs\n"
                "converted numpy.ndarray shape is: {}. Expected {}".format(
                    starts_ends.shape, (len(labels), 2)))

        if samplerate <= 0:
            # IDEA: Support negative samplerate?
            raise ValueError("samplerate <= 0 not supported")
        else:
            if np.any(starts_ends[:, 1] - starts_ends[:, 0] <= 0):
                raise ValueError("(ends - starts) should be > 0 for all pairs")
            # sort primarily by starts, and secondarily by ends
            sort_idx = np.lexsort(np.split(starts_ends[..., ::-1].T, 2))

        if not sort_idx.shape[0] == 1:
            # something has gone horribly wrong
            raise RuntimeError(
                "sort_idx has an unexpected shape: {}\nShould have been {}".
                format(sort_idx.shape, (1, ) + sort_idx.shape[1:]))

        sort_idx = sort_idx[0, :]  # shape in dim-0 **should** always be 1
        self._starts_ends = starts_ends[sort_idx, ...]

        if isinstance(labels, np.ndarray):
            self.labels = labels[sort_idx, ...]
        else:
            self.labels = tuple(labels[i] for i in sort_idx)

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
    def samplerate_as(self, new_samplerate):
        """ Temporarily change to a different samplerate within context

        To be used with a `with` clause, and supports nesting of such clauses.

        if `new_samplerate` is `None`, the samplerate will remain as the
        contextually most recent non `None` samplerate.

        This can be used to get `starts_ends` as if they were calculated with
        different samplerate than original. Within a nested `with` clause,
        the samplerate from the most recent clause will be used.

        For example, for segment with `starts_ends` [[1, 5]] at samplerate 1,
        when calculated in context of `new_samplerate = 2`, the `starts_ends`
        will be [[2, 10]].
        """
        old_sr = self.samplerate
        self._samplerate = old_sr if new_samplerate is None else new_samplerate
        try:
            yield
        finally:
            self._samplerate = old_sr

    def labels_at(self, ends, samplerate=None, default_label=()):
        """ TODO: [ ] Proper Dox

        if `samplerate` is `None`, it is assumed that `ends` are at the same
        `samplerate` as our contextually most recent one. See `samplerate_as`
        """
        if not isinstance(ends, Iterable):
            ends = [ends]

        ends = np.array(ends)

        with self.samplerate_as(samplerate):
            se = self.starts_ends

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

    def _infer_and_get_filled_default_labels(self, shape):
        if self.labels.dtype == np.object:
            return np.array([None for _ in range(len(shape[0]))])
        else:
            return np.zeros(
                shape=((shape[0], ) + self.labels.shape[1:]),
                dtype=self.labels.dtype)

    def labels_at(self, ends, samplerate=None, default_label='auto'):
        """ TODO: [ ] Proper Dox

        if `samplerate` is `None`, it is assumed that `ends` are at the same
        `samplerate` as our contextually most recent one. See `samplerate_as`
        """
        if not isinstance(ends, Iterable):
            ends = [ends]

        ends = np.array(ends)

        with self.samplerate_as(samplerate):
            se = self.starts_ends

        se = np.round(se, 10)  # To avoid issues with floating points
        # Yes, it looks arbitrary. Check SequenceLabels.labels_at(...)
        #
        # FIXME: It should not be the case anyway. All the trouble to support
        #       arbitrary self.orig_samplerate and samplerate
        #

        # all ends that are within the segments
        endings = se[:, 1]
        maxend = endings.max()
        minstart = se[:, 0].min()
        endswithin = (ends > minstart) & (ends <= maxend)

        # find indices of the labels for ends that are within the segments
        within_labelidx = np.searchsorted(
            endings, ends[endswithin], side='left')

        if endswithin.sum() == len(ends):
            # all ends are within

            # pick the labels at those indices, and return
            return self.labels[within_labelidx, ...]

        elif default_label == 'auto':
            # some ends are outside and a default label is not provided

            # a default label will be inferred from the existing self.labels
            # We construct the numpy array with default label for all ends
            res = self._infer_and_get_filled_default_labels(ends.shape)

            # then fill it up with found labels where ends are within
            res[endswithin] = self.labels[within_labelidx, ...]

            return res
        else:
            # provided default_label will be inserted for ends which are outside

            label_idx = np.ones_like(ends, dtype=np.int) * -1
            label_idx[endswithin] = within_labelidx

            result = []
            for li in label_idx:
                if li < 0:  # default_label
                    result.append(default_label)
                else:
                    try:
                        result.append(self.labels[li, ...])
                    except IndexError as e:
                        print(li)
                        raise e

            return result


def times_for_labelsat(total_duration_sec, samplerate, hop_sec, win_sec):
    # NOTE: all the samplerate multiplication cuz float is fucking AWESOME
    hop_len = int(hop_sec * samplerate)
    win_len = int(win_sec * samplerate)
    nsamples = int(total_duration_sec * samplerate)

    return samples_for_labelsat(nsamples, hop_len, win_len) / samplerate


def samples_for_labelsat(nsamples, hop_len, win_len):
    nframes = 1 + (nsamples - win_len) // hop_len
    frames_idx = np.arange(nframes)

    samples_out = (frames_idx * hop_len) + (win_len // 2)

    return samples_out
