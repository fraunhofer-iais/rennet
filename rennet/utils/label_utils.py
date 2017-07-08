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
    """Base class for working with labels for a sequence.

    By default the samplerate is 1, but a default one can be set at the time
    of instantiating. The samplerate should reflect the one used in calculating
    the starts_ends.

    Segments will get sorted based on `starts` (primary; `ends` secondary).

    Supports normal indexing and slicing, but the returned value will be another
    instance of the SequenceLabels class (or the relevant starts_ends, labels
    and orig_samplerate, when being called from a subclass)

    When iterated over, the returned values are a `zip` of `starts_ends` and
    `labels` for each segment.
    """
    __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate')
    """To save memory, maybe? I just wanted to learn about them.
    Include at least ``__slots__ = ()`` if you want to keep the functionality in
    a subclass.
    """

    def __init__(self, starts_ends, labels, samplerate=1):
        """Initialize a SequenceLabels instance with starts_ends and labels"""
        # TODO: [ ] Add dox, at least the params and attributes

        if any(not isinstance(x, Iterable) for x in [starts_ends, labels]):
            raise TypeError("starts_ends and labels should be Iterable")
        if len(starts_ends) != len(labels):
            raise AssertionError("starts_ends and labels mismatch in length")

        labels = np.array(labels)
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
        self.labels = labels[sort_idx, ...]

        self._orig_samplerate = samplerate
        self._samplerate = samplerate

    @property
    def samplerate(self):
        """float or int: The current samplerate of `starts_ends`.

        Note
        ----
        The current samplerate can be changed within a `samplerate_as` context.
        But that will also impact how the `starts_ends` are calculated.
        """
        return self._samplerate

    @property
    def orig_samplerate(self):
        """float or int: The original samplerate of `starts_ends`.

        Note
        ----
        The effective samplerate can be changed within a `samplerate_as` context.
        Here as a property to discourage from changing after initialization.
        """
        return self._orig_samplerate

    @property
    def starts_ends(self):
        """ndarray: `starts_ends` of the `labels`, calculated with contextually
        the most recent non-`None` samplerate. See also `samplerate_as`.
        """
        sr = self.samplerate  # contextually most recent and valid samplerate
        osr = self.orig_samplerate
        if sr >= osr and sr % osr == 0:
            # avoid definitely floating a potential int
            # will still return float if any of the three is float
            # worth a try I guess
            return self._starts_ends * (sr // osr)
        else:
            return self._starts_ends * (sr / osr)

    @contextmanager
    def samplerate_as(self, new_samplerate):
        """Temporarily change to a different samplerate within context.

        To be used with a `with` clause, and supports nesting of such clauses.
        Within a nested `with` clause, the samplerate from the most recent
        clause will be used.

        This can be used to get `starts_ends` as if they were calculated with
        different samplerate than the original.

        Note
        ----
        EXCEPT for one for indexing/slicing, all methods honour the contextually
        most recent and valid samplerate in their calculations. New instances
        created on indexing/slicing are always created with the original samplerate.

        Parameters
        ----------
        new_samplerate : float or int or None
            The new samplerate with which `starts_ends` will be calculated while
            within the `with` clause. if `None`, the samplerate will remain as
            the contextually most recent non-`None` value.

        Raises
        ------
        ValueError
            If `new_samplerate` <= 0

        Example
        -------
        For example, for segment with `starts_ends` [[1., 5.]] at samplerate 1,
        when calculated in context of `new_samplerate = 2`, the `starts_ends`
        will be [[2., 10.]].
        """
        old_sr = self.samplerate
        new_sr = old_sr if new_samplerate is None else new_samplerate
        if new_sr <= 0: raise ValueError("new_samplerate <=0 not supported")

        self._samplerate = new_sr
        try:
            yield
        finally:
            self._samplerate = old_sr

    def _flattened_indices(self, return_bins=False):
        """Calculate indices of the labels that form the flattened labels.

        Flattened means, there is 1 and only 1 "label" for each time-step within
        the min-start and max-end.
        """
        # TODO: Proper dox; add params and returns

        se = self.starts_ends

        if np.all(se[1:, 0] == se[:-1, 1]):  # already flat
            bins = np.zeros(len(se) + 1, dtype=se.dtype)
            bins[:-1] = se[:, 0]
            bins[-1] = se[-1, 1]

            labels_indices = [(i, ) for i in range(len(se))]
        else:
            # `numpy.unique` also sorts the (flattened) array
            bins, sorting_indices = np.unique(se, return_inverse=True)

            sorting_indices = sorting_indices.reshape(-1, 2)  # un-flatten

            labels_indices = [tuple()] * (len(bins) - 1)
            for j, (s, e) in enumerate(sorting_indices):
                # `(e - s)` is small but > 0, usually 1
                # `s` may also repeat, hence can't do fancy `numpy` w/ readability
                for i in range(s, e):
                    labels_indices[i] += (j, )

        if return_bins:
            # return as bins for `numpy.digitize`
            return bins, labels_indices
        else:
            # return as `starts_ends` for `ContiguousSequenceLabels`
            return np.stack((bins[:-1], bins[1:]), axis=1), labels_indices

    def labels_at(self, ends, samplerate=None, default_label=(), rounded=10):
        """ TODO: [ ] Proper Dox

        if `samplerate` is `None`, it is assumed that `ends` are at the same
        `samplerate` as our contextually most recent one. See `samplerate_as`
        """
        if not isinstance(ends, Iterable):
            ends = [ends]
        ends = np.array(ends)

        with self.samplerate_as(samplerate):
            bins, labels_idx = self._flattened_indices(return_bins=True)

        bins = np.round(bins, rounded)  # floating point comparison issues

        # include right edge, not left, because we only know about what happened
        # in the `(1/_orig_samplerate)` seconds finishing at an `end`
        bin_idx = np.digitize(ends, bins, right=True)

        # construct labels for only the unique bin_idx, repackage when returning
        unique_bin_idx, bin_idx = np.unique(bin_idx, return_inverse=True)

        unique_res_labels = np.empty(len(unique_bin_idx), dtype=np.object)
        unique_res_labels.fill(default_label)
        for i, idx in enumerate(unique_bin_idx):
            if idx != 0 and idx != len(bins):  # if not outside bins

                # labels for bin_idx == 1 are at labels_idx[0]
                l = labels_idx[idx - 1]
                if len(l) == 1:
                    unique_res_labels[i] = (self.labels[l[0]], )
                elif len(l) > 1:
                    unique_res_labels[i] = tuple(self.labels[list(l)])
                else:
                    unique_res_labels[i] = default_label

        return unique_res_labels[bin_idx, ...]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        se = self._starts_ends[idx, ...]
        l = self.labels[idx, ...]

        if len(se.shape) == 1:  # case with only one segment
            se = se[None, ...]
            l = l[None, ...]

        if self.__class__ is SequenceLabels:
            return self.__class__(se, l, self.orig_samplerate)
        else:
            return se, l, self.orig_samplerate

    def __iter__(self):
        return zip(self.starts_ends, self.labels)

    def __str__(self):
        s = ".".join((self.__module__.split('.')[-1], self.__class__.__name__))
        s += " with sample rate {}\n".format(self.samplerate)
        s += "{:8} - {:8} : {}\n".format("Start", "End", "Label")
        s += "\n".join("{:<8.4f} - {:<8.4f} : {}".format(s, e, str(l))
                       for (s, e), l in self)
        return s

    # TODO: [ ] Import from ELAN
    # TODO: [ ] Export to ELAN
    # TODO: [ ] Import from mpeg7
    # TODO: [ ] Export to mpeg7


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
        if not np.all(self.starts_ends[1:, 0] == self.starts_ends[:-1, 1]):
            msg = "All ends should be the starts of the next segment, except in the case of the last segment."
            msg += "\nEvery time-step should belong to 1 and only 1 segment."
            msg += "\nNo duplicate or missing segments allowed between min-start and max-end"
            raise AssertionError(msg)

    def _infer_and_get_filled_default_labels(self, shape):
        if self.labels.dtype == np.object:
            return np.array([None for _ in range(len(shape[0]))])
        else:
            return np.zeros(
                shape=((shape[0], ) + self.labels.shape[1:]),
                dtype=self.labels.dtype)

    def labels_at(self,
                  ends,
                  samplerate=None,
                  default_label='auto',
                  rounded=10):
        """ TODO: [ ] Proper Dox

        if `samplerate` is `None`, it is assumed that `ends` are at the same
        `samplerate` as our contextually most recent one. See `samplerate_as`
        """
        if not isinstance(ends, Iterable):
            ends = [ends]

        ends = np.array(ends)

        with self.samplerate_as(samplerate):
            se = self.starts_ends

        se = np.round(se, rounded)  # To avoid issues with floating points

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

    def __getitem__(self, idx):
        res = super(ContiguousSequenceLabels, self).__getitem__(idx)
        if self.__class__ is ContiguousSequenceLabels:
            return self.__class__(*res)
        else:
            return res


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
