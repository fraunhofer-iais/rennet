"""
@motjuste
Created: 26-08-2016

Utilities for working with labels
"""
from __future__ import print_function, division
import numpy as np
from collections import Iterable
from contextlib import contextmanager
from itertools import groupby


class SequenceLabels(object):
    """Base class for working with labels for a sequence.

    By default the samplerate is 1, but a default one can be set at the time
    of instantiating. The samplerate should reflect the one used in calculating
    the starts_ends.

    Segments will get sorted based on `starts` (primary; `ends` secondary).

    Supports normal indexing and slicing, but the returned value will be another
    instance of the SequenceLabels class (or the relevant starts_ends, labels
    and current samplerate, when being called from a subclass)

    When iterated over, the returned values are a `zip` of `starts_ends` and
    `labels` for each segment.
    """
    __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate',
                 '_minstart_at_orig_sr', )

    # To save memory, maybe? I just wanted to learn about them.
    # NOTE: Add at least ``__slots__ = ()`` at the top if you want to keep the functionality in a subclass.

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
                raise ValueError("(end - start) should be > 0 for all pairs")
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

        self._minstart_at_orig_sr = self._starts_ends.item(0)  # min-start

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

    @staticmethod
    def _convert_samplerate(value, from_samplerate, to_samplerate):
        """ Convert a value from_samplerate to_samplerate

        Tries to keep the return value int when possible and foreseen.

        Parameters
        ----------
        value: ndarray or float or int
            The value whose samplerate has to be changed.
        from_samplerate: float or int, > 0
            The samplerate of value.
        to_samplerate: float or int, > 0
            The samplerate the value is to be converted to.

        Raises
        ------
        ValueError: When from_samplerate <= 0 or to_samplerate <= 0
        """
        if to_samplerate <= 0 or from_samplerate <= 0:
            raise ValueError(
                "samplerates <=0 not supported: from_samplerate= {}, to_samplerate= {}".
                format(from_samplerate, to_samplerate))

        if to_samplerate == from_samplerate or (
                not isinstance(value, np.ndarray) and value == 0):
            return value

        if to_samplerate > from_samplerate and to_samplerate % from_samplerate == 0:
            # avoid definitely floating a potential int
            # will still return float if any of the three is float
            # worth a try I guess
            return value * (to_samplerate // from_samplerate)
        else:
            return value * (to_samplerate / from_samplerate)

    @property
    def min_start(self):
        """ float or int: Minimum start of starts_ends at the current samplerate.

        Effectively, the start time-point of the first label, when all are sorted
        based on starts as primary key, and ends as secondary key.
        """
        # self._minstart_at_orig_sr is always at self._orig_samplerate
        return self._convert_samplerate(
            self._minstart_at_orig_sr,
            from_samplerate=self._orig_samplerate,
            to_samplerate=self._samplerate, )

    @property
    def max_end(self):
        """ float or int: Maximum end of starts_ends at the current samplerate.

        Effectively, the end time-point of the last label, when all are sorted
        based on starts as primary key, and ends as secondary key.
        """
        return self.starts_ends.item(-1)

    @property
    def starts_ends(self):
        """ndarray: `starts_ends` of the `labels`, calculated with contextually
        the most recent non-`None` samplerate. See also `samplerate_as`.

        self._starts_ends is stored at self._orig_samplerate and never modified.
        """
        starts_ends = self._starts_ends
        ominstart = starts_ends.item(0)
        if self._minstart_at_orig_sr != ominstart:
            starts_ends = starts_ends - (ominstart - self._minstart_at_orig_sr)

        return self._convert_samplerate(
            starts_ends,
            from_samplerate=self._orig_samplerate,
            to_samplerate=self._samplerate, )

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
        All methods honour the contextually most recent and valid samplerate in their calculations.
        New instances created on indexing/slicing are also created with the recent samplerate.

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
        old_sr = self._samplerate
        new_sr = old_sr if new_samplerate is None else new_samplerate
        if new_sr <= 0: raise ValueError("new_samplerate <=0 not supported")

        self._samplerate = new_sr
        try:
            yield
        finally:
            self._samplerate = old_sr

    @contextmanager
    def min_start_as(self, new_start, samplerate=None):
        """Temporarily shift all `starts_ends` to start at `new_start`.

        To be used with a `with` clause, and supports nesting of such clauses.
        Within a nested `with` clause, the `min_start` from the most recent
        clause will be used.

        This can be used to get `starts_ends` as if they started from a different
        `min_start` than the original.

        If `samplerate` is `None`, then it is assumed that the samplerate of
        `new_start` is the same as the contextually most recently valid one.

        If a `samplerate` is provided, the `starts_ends` are calculated based on
        this samplerate and the shifting is done thereafter. It is equivalent to
        applying the `samplerate` in a `with i.samplerate_as(samplerate):` clause
        and then doing the shifting to `new_start`.

        Note
        ----
        All methods honour the contextually most recent `min_start` in their calculations.
        New instances created on indexing/slicing are also created with the recent `min_start`.

        Parameters
        ----------
        new_start : float or int
            The new min_start with which `starts_ends` will be shifted to while
            within the `with` clause.

        samplerate: float or int or None
            The samplerate of new_start. If `None` (default), it is assumed that
            `new_start` at the contextually most recent samplerate.

        Raises
        ------
        ValueError
            If `samplerate` <= 0

        Example
        -------
        For example, for segment with `starts_ends` [[1., 5.]] at samplerate 1,
        when calculated in context of `new_start = 2`, the `starts_ends`
        will be [[2., 6.]].
        """
        old_start = self._minstart_at_orig_sr
        with self.samplerate_as(samplerate):
            # context needed to handle provided samplerate
            # self._samplerate will then have the valid samplerate
            # e.g. if provided value for samplerate is None then
            # self._samplerate will be the contextually most recently valid one

            # the _minstart_at_orig_sr is always at self._orig_samplerate
            self._minstart_at_orig_sr = self._convert_samplerate(
                new_start,
                from_samplerate=self._samplerate,
                to_samplerate=self._orig_samplerate, )
            try:
                yield
            finally:
                self._minstart_at_orig_sr = old_start

    # NOTE: A context manager like max_end_as has been avoided due to complications
    # resulting from shifting both the start and end at the same time,
    # which will definitely lead to change in samplerate (and I don't want to implement it), or
    # needs me to do my PhD first.

    def _flattened_indices(self, return_bins=False):
        """Calculate indices of the labels that form the flattened labels.

        Flattened means, there is 1 and only 1 "label" for each time-step within
        the min-start and max-end. No less, no more.

        That is, all time-steps between min-start and max-end are accounted for,
        even if with an empty `tuple()`.

        Returns empty `tuple()` for start-end pairs for which no labels can be inferred.
        """
        # TODO: Proper dox; add params and returns

        se = self.starts_ends

        if np.any(se[1:, 0] != se[:-1, 1]):  # not flat
            # `numpy.unique` also sorts the (flattened) array
            bins, sorting_indices = np.unique(se, return_inverse=True)

            sorting_indices = sorting_indices.reshape(-1, 2)  # un-flatten

            labels_indices = [tuple()] * (len(bins) - 1)
            for j, (s, e) in enumerate(sorting_indices):
                # `(e - s)` is small but > 0, usually 1
                # `s` may also repeat, hence can't do fancy `numpy` w/ readability
                for i in range(s, e):
                    labels_indices[i] += (j, )
        else:  # already flat
            bins = np.zeros(len(se) + 1, dtype=se.dtype)
            bins[:-1] = se[:, 0]
            bins[-1] = se[-1, 1]

            labels_indices = [(i, ) for i in range(len(se))]

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
        if not isinstance(ends, np.ndarray):
            ends = np.array(ends)

        with self.samplerate_as(samplerate):
            bins, labels_idx = self._flattened_indices(return_bins=True)

        if ends.dtype != np.int or bins.dtype != np.int:
            # floating point comparison issues
            bins = np.round(bins, rounded)
            ends = np.round(ends, rounded)

        # We only know about what happened in the `(1/_orig_samplerate)` seconds starting at an `end`
        # Hence choose side='right'.
        # ends outside bins will have value 0 or len(bins)
        bin_idx = np.searchsorted(bins, ends, side='right')

        # construct labels for only the unique bin_idx, repackage when returning
        unique_bin_idx, bin_idx = np.unique(bin_idx, return_inverse=True)

        unique_res_labels = np.empty(len(unique_bin_idx), dtype=np.object)
        unique_res_labels.fill(default_label)
        for i, idx in enumerate(unique_bin_idx):
            if idx != 0 and idx != len(bins):  # if not outside bins

                # labels for bin_idx == 1 are at labels_idx[0]
                l = labels_idx[idx - 1]
                if len(l) == 1:
                    unique_res_labels[i] = (self.labels[l[0], ...], )
                elif len(l) > 1:
                    unique_res_labels[i] = tuple(self.labels[l, ...])
                # else: it is prefilled with default_label

        return unique_res_labels[bin_idx, ...]

    @classmethod
    def from_dense_labels(  # pylint: disable=too-many-arguments
            cls,
            labels,
            groupby_keyfn=None,
            keep='both',
            min_start=0,
            samplerate=1,
            **kwargs):
        """ Create SequenceLabels instance from dense list of labels.

        Parameters
        ----------
        labels: array_like
            Dense labels.
        groupby_keyfn: key function
            The key function that itertools.groupby will use to make groups.
            See example below.
        keep: 'both', 'keys', 'labels'
            - 'keys': keep only the result of groupby_keyfn(label) in final labels.
            - 'labels': keep only list of consecutive labels that have the same groupby_keyfn(label) result.
            - 'both': keep both in a tuple.
        min_start: int or float
            The start of the first segmentation. By default zero.
            The value should be at the same samplerate as provided samplerate.
        samplerate: int
            samplerate at which the labels were taken.
            That is, how many labels occur in 1 second.
            The start time can be changed by setting the min_start.

        Examples
        --------
        >>> labels = [[0.2, 0.8], [0.3, 0.7], [0.51, 0.49], [0.55, 0.45], [0.40, 0.60]]
        >>> sr = 2000
        >>> ms = 0.016 * sr  # 32
        >>> key = np.argmax
        >>> s = SequenceLabels.from_dense_labels(labels, key, keep='keys', min_start=ms, samplerate=sr)
        >>> list(s)
        [(array([ 32.,  34.]), 1), (array([ 34.,  36.]), 0), (array([ 36.,  37.]), 1)]
        >>>
        >>> list(SequenceLabels.from_dense_labels(labels, key, keep='labels', min_start=ms, samplerate=sr))
        [(array([ 32.,  34.]), ([0.2, 0.8], [0.3, 0.7])),
         (array([ 34.,  36.]), ([0.51, 0.49], [0.55, 0.45])),
         (array([ 36.,  37.]), ([0.4, 0.6],))]
        >>>
        >>> list(SequenceLabels.from_dense_labels(labels, key, keep='both', min_start=ms, samplerate=sr))
        [(array([ 32.,  34.]), array([1, ([0.2, 0.8], [0.3, 0.7])], dtype=object)),
         (array([ 34.,  36.]), array([0, ([0.51, 0.49], [0.55, 0.45])], dtype=object)),
         (array([ 36.,  37.]), array([1, ([0.4, 0.6],)], dtype=object))]
        """
        keep = str(keep).lower()
        if keep not in ['both', 'keys', 'labels']:
            raise ValueError("Unsupported value for keep {:r}. ".format(keep) +\
                             "Accepted values {'both', 'keys', 'labels'}.")

        if not isinstance(samplerate, int):
            raise TypeError('samplerate should be of type int, not {}'.format(
                type(samplerate)))
        if samplerate <= 0:
            raise ValueError(
                'samplerate should be >= 0, not {}'.format(samplerate))

        keylabels = []
        bins = [0]
        for k, it in groupby(labels, groupby_keyfn):
            lit = tuple(it)
            keylabels.append((k, lit))
            bins.append(bins[-1] + len(lit))

        bins = np.array(bins) + min_start
        se = np.stack((bins[:-1], bins[1:]), axis=1)

        if keep == 'both':
            keylabels = np.array(keylabels, dtype=np.object)
        else:
            label_keys, label_list = list(zip(*keylabels))
            if keep == 'keys':
                keylabels = label_keys
            elif keep == 'labels':
                keylabels = label_list

        if cls == SequenceLabels:
            return cls(se, keylabels, samplerate)
        else:
            return se, keylabels, samplerate, kwargs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        se = self.starts_ends[idx, ...]
        l = self.labels[idx, ...]

        if len(se.shape) == 1:  # case with only one segment
            se = se[None, ...]
            l = l[None, ...]

        if self.__class__ is SequenceLabels:
            return self.__class__(se, l, self.orig_samplerate)
        else:
            return se, l, self._samplerate

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
    # __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate',
    #              '_minstart_at_orig_sr', )
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(ContiguousSequenceLabels, self).__init__(*args, **kwargs)
        # the starts_ends were sorted in __init__ on starts
        if not np.all(self.starts_ends[1:, 0] == self.starts_ends[:-1, 1]):
            msg = "All ends should be the starts of the next segment, except in the case of the last segment."
            msg += "\nEvery time-step should belong to 1 and only 1 segment."
            msg += "\nNo duplicate or missing segments allowed between min-start and max-end"
            raise AssertionError(msg)

    def labels_at(self,
                  ends,
                  samplerate=None,
                  default_label='zeros',
                  rounded=10):
        """ Get labels at ends.

        if `samplerate` is `None`, it is assumed that `ends` are at the same
        `samplerate` as our contextually most recent one. See `samplerate_as`

        TODO: [ ] Proper Dox
        """
        if not isinstance(ends, Iterable):
            ends = [ends]
        if not isinstance(ends, np.ndarray):
            ends = np.array(ends)

        with self.samplerate_as(samplerate):
            se = self.starts_ends
            bins = np.append(se[:, 0], se[-1, -1])

        if ends.dtype != np.int or bins.dtype != np.int:
            # floating point comparison issues
            bins = np.round(bins, rounded)
            ends = np.round(ends, rounded)

        # We only know about what happened in the `(1/_orig_samplerate)` seconds starting at an `end`
        # Hence choose side='right'.
        # np.digitize is slower!!!
        bin_idx = np.searchsorted(bins, ends, side='right')

        # ends which are not within the bins will be either 0 or len(bins)
        bin_idx_outside = (bin_idx == 0) | (bin_idx == len(bins))

        # label for bin_idx == 1 is at labels[0]
        bin_idx -= 1
        if np.any(bin_idx_outside):
            # there are some ends outside the bins
            if default_label == 'raise':
                with self.samplerate_as(samplerate):
                    msg = "Some ends are outside the segments and default_label has been chosen to be 'raise'. "+\
                        "Choose an appropriate default_label, or ammend the provided ends to be in range ="+\
                        " ({}, {}] at samplerate {}".format(bins[0], bins[-1], self.samplerate)
                    raise KeyError(msg)

            bin_idx_within = np.invert(bin_idx_outside)
            res = np.zeros(
                shape=(len(bin_idx), ) + self.labels.shape[1:],
                dtype=self.labels.dtype, )
            res[bin_idx_within] = self.labels[bin_idx[bin_idx_within], ...]

            if default_label == 'zeros':
                # IDEA: be more smart about handling some of the custom default labels.
                pass
            elif default_label == 'ones':
                res[bin_idx_outside] = 1
            elif type(default_label) == res.dtype:
                # IDEA: provide way to handle numpy.ndarray type of default_label
                # if it has the right shape (and maybe type as well)

                # IDEA: provide way to fill with a default_label of different dtype
                # perhaps by casting to np.object
                # may require extra parameter like force_fill=True or something

                # The user is probably asking to fill the array with default_label where ends are outside
                res[bin_idx_outside] = default_label
            else:
                # IDEA: provide more options for default_label, like, Nones, etc.

                # fallback case to handle a provided default_label
                res = res.tolist()
                for oi in np.where(bin_idx_outside)[0]:
                    res[oi] = default_label

            return res
        else:
            # all ends are within bins
            return self.labels[bin_idx, ...]

    @classmethod
    def from_dense_labels(  # pylint: disable=too-many-arguments
            cls,
            labels,
            groupby_keyfn=None,
            keep='both',
            min_start=0,
            samplerate=1,
            **kwargs):
        """ Create SequenceLabels instance from dense list of labels.

        Parameters
        ----------
        labels: array_like
            Dense labels.
        groupby_keyfn: key function
            The key function that itertools.groupby will use to make groups.
            See example below.
        keep: 'both', 'keys', 'labels'
            - 'keys': keep only the result of groupby_keyfn(label) in final labels.
            - 'labels': keep only list of consecutive labels that have the same groupby_keyfn(label) result.
            - 'both': keep both in a tuple.
        min_start: int or float
            The start of the first segmentation. By default zero.
            The value should be at the same samplerate as provided samplerate.
        samplerate: int
            samplerate at which the labels were taken.
            That is, how many labels occur in 1 second.
            The start time can be changed by setting the min_start.

        Examples
        --------
        >>> labels = [[0.2, 0.8], [0.3, 0.7], [0.51, 0.49], [0.55, 0.45], [0.40, 0.60]]
        >>> sr = 2000
        >>> ms = 0.016 * sr  # 32
        >>> key = np.argmax
        >>> s = SequenceLabels.from_dense_labels(labels, key, keep='keys', min_start=ms, samplerate=sr)
        >>> list(s)
        [(array([ 32.,  34.]), 1), (array([ 34.,  36.]), 0), (array([ 36.,  37.]), 1)]
        >>>
        >>> list(SequenceLabels.from_dense_labels(labels, key, keep='labels', min_start=ms, samplerate=sr))
        [(array([ 32.,  34.]), ([0.2, 0.8], [0.3, 0.7])),
         (array([ 34.,  36.]), ([0.51, 0.49], [0.55, 0.45])),
         (array([ 36.,  37.]), ([0.4, 0.6],))]
        >>>
        >>> list(SequenceLabels.from_dense_labels(labels, key, keep='both', min_start=ms, samplerate=sr))
        [(array([ 32.,  34.]), array([1, ([0.2, 0.8], [0.3, 0.7])], dtype=object)),
         (array([ 34.,  36.]), array([0, ([0.51, 0.49], [0.55, 0.45])], dtype=object)),
         (array([ 36.,  37.]), array([1, ([0.4, 0.6],)], dtype=object))]
        """
        params = super(ContiguousSequenceLabels, cls).from_dense_labels(
            labels, groupby_keyfn, keep, min_start, samplerate, **kwargs)
        if cls == ContiguousSequenceLabels:
            return cls(*params[:-1])
        else:
            return params

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


def viterbi_smoothing(obs, init, tran, amin=1e-15):
    obs = np.log(np.maximum(amin, obs))  # pylint: disable=no-member
    init = np.log(np.maximum(amin, init))  # pylint: disable=no-member
    tran = np.log(np.maximum(amin, tran))  # pylint: disable=no-member

    backpt = np.ones_like(obs, dtype=np.int) * -1
    trellis_last = init + obs[0, ...]
    for t in range(1, len(obs)):
        x = trellis_last[None, ...] + tran
        backpt[t, ...] = np.argmax(x, axis=1)
        trellis_last = np.max(x, axis=1) + obs[t, ...]

    tokens = np.ones(shape=len(obs), dtype=np.int) * -1
    tokens[-1] = trellis_last.argmax()
    for t in range(len(obs) - 2, -1, -1):
        tokens[t] = backpt[t + 1, tokens[t + 1]]

    return tokens
