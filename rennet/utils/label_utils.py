#  Copyright 2018 Fraunhofer IAIS. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Utilities for working with labels

@motjuste
Created: 26-08-2016
"""
from __future__ import print_function, division, absolute_import
from six.moves import zip, range
import numpy as np
from collections import Iterable, OrderedDict
from contextlib import contextmanager
from itertools import groupby
from os.path import abspath
import sys
import warnings

from pympi import Eaf

from .. import __version__ as rennet_version
from .py_utils import BaseSlotsOnlyClass
from .np_utils import normalize_confusion_matrix, confusion_matrix_forcategorical
from .mpeg7_utils import parse_mpeg7


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
    __slots__ = (
        '_starts_ends',
        'labels',
        '_orig_samplerate',
        '_samplerate',
        '_minstart_at_orig_sr',
    )

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
                    starts_ends.shape, (len(labels), 2)
                )
            )

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
                "sort_idx has an unexpected shape: {}\nShould have been {}".format(
                    sort_idx.shape, (1, ) + sort_idx.shape[1:]
                )
            )

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
                format(from_samplerate, to_samplerate)
            )

        if to_samplerate == from_samplerate or (
            not isinstance(value, np.ndarray) and value == 0
        ):
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
            to_samplerate=self._samplerate,
        )

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
            to_samplerate=self._samplerate,
        )

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
                to_samplerate=self._orig_samplerate,
            )
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

        NOTE: if the callee `cls` is not SequenceLabels, then no class is instantiated.
        It will be the responsibility of the callee (probably a child class) to create
        the appropriate instance of it's class.

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
            raise TypeError(
                'samplerate should be of type int, not {}'.format(type(samplerate))
            )
        if samplerate <= 0:
            raise ValueError('samplerate should be >= 0, not {}'.format(samplerate))

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

        # create the sub-SequenceLabel instance with the contextually correct samplerate
        # and *NOT* the original samplerate
        if self.__class__ is SequenceLabels:
            return self.__class__(se, l, self.samplerate)
        else:
            # some child class
            # let's honor kwargs, they should too
            return se, l, self.samplerate

    def __iter__(self):
        # NOTE: Yes, it is known that there is a disparity between __getitem__
        # returning new instance of SequenceLabels, and __iter__ returning
        # zipped starts_ends and labels.
        # It was done so to meet actual usage patterns.
        # iterating over the individual pairs (start_end, label) was far more common.
        # FIXME: Fix the disparity between __getitem__ and __iter__
        return zip(self.starts_ends, self.labels)

    def __str__(self):
        s = ".".join((self.__module__.split('.')[-1], self.__class__.__name__))
        s += " with sample rate {}\n".format(self.samplerate)
        s += "{:8} - {:8} : {}\n".format("Start", "End", "Label")
        s += "\n".join(
            "{:<8.4f} - {:<8.4f} : {}".format(s, e, str(l)) for (s, e), l in self
        )
        return s

    @classmethod
    def from_mpeg7(cls, filepath, use_tags='ns', **kwargs):
        """ Create instance of SequenceLabels from an mpeg7 annotation file.

        NOTE: if the callee `cls` is not SequenceLabels, then no class is instantiated.
        It will be the responsibility of the callee (probably a child class) to create
        the appropriate instance of it's class.

        NOTE: Supported use_tags: "ns" (default), "mpeg7".

        Parameters
        ----------
        filepath: path to the ELAN file
        use_tags: options: 'ns' or 'mpeg7'
            Check `rennet.utils.mpeg7_utils`.
        kwargs: unused, present for proper sub-classing citizenship.

        Returns
        -------
        - if callee `cls` is not `SequenceLabels` (probably a child class):
            starts_ends: numpy.ndarray of numbers, of shape `(num_annotations_read, 2)`.
            labels: list of MPEG7AnnotationInfo objects, of length `num_annotations_read`.
            samplerate: int (most likely 1000, due to limits of `pympi`), the samplerate
            **kwargs: passed through keyword arguments `**kwargs`
        - else:
            instance of SequenceLabels

        Raises
        ------
        RuntimeError: if not annotations are found in the given file.
        ValueError: Check `rennet.utils.mpeg7_utils`.
        """
        # se, sr, sids, gen, gn, conf, trn = parse_mpeg7(filepath, use_tags=use_tags)
        filepath = abspath(filepath)
        parsed = parse_mpeg7(filepath, use_tags=use_tags)
        starts_ends, samplerate = parsed[:2]

        if len(starts_ends) == 0:
            raise RuntimeError(
                "No Annotations were found from file {}.\n".format(filepath) + \
                "Check `use_tags` parameter for `mpeg7_utils.parse_mpeg7` "+\
                "and pass appropriate one as keyword argument to this function.\n"+\
                "Options: 'ns' (default) and 'mpeg7'"
                )

        labels = [
            MPEG7AnnotationInfo(
                speakerid=sid,
                gender=gen,
                givenname=gn,
                confidence=conf,
                content=trn,
            ) for sid, gen, gn, conf, trn in zip(*parsed[2:])
        ]

        if cls == SequenceLabels:
            return cls(starts_ends, labels, samplerate)
        else:
            # some child class
            # let's honor kwargs, they should too
            return starts_ends, labels, samplerate, kwargs

    @classmethod
    def from_eaf(cls, filepath, tiers=(), **kwargs):
        """ Create instance of SequenceLabels from an ELAN annotation file.

        NOTE: Not all features of ELAN files are supported. For example:
        - Only the aligned annotations are read.
        - No attempt is made to read any external refernces, linked files, etc.
        - Multi-level tier heirarchies are not respected.

        NOTE: Annotations of duration <= 0 will be ignored.

        Parameters
        ----------
        filepath: path to the ELAN file
        tiers: list or tuple of strings or unary callable
            list or tuple of tier names (as strings) to specify what tiers to be read.
            By default, this is an empty tuple (or list), and all tiers will be read.
            If it is an unary callable (i.e. taking one argument), the function will be
            used as the predicated to filter which tiers should be kept.
        kwargs: unused, present for proper sub-classing citizenship

        Returns
        -------
        - if callee `cls` is not `SequenceLabels` (probably a child class):
            starts_ends: numpy.ndarray of numbers, of shape `(num_annotations_read, 2)`.
            labels: list of EafAnnotationInfo objects, of length `num_annotations_read`.
            samplerate: int (most likely 1000, due to limits of `pympi`), the samplerate
            **kwargs: passed through keyword arguments `**kwargs`
        - else:
            instance of SequenceLabels

        Raises
        ------
        TypeError: if `tiers` is neither a tuple nor a list (of strings).
        KeyError: if any of the specified tier names are not available in the given file.
        RuntimeError: if no tiers are found, or if all tiers are empty
        """
        filepath = abspath(filepath)
        eaf = Eaf(file_path=filepath)

        # FIXME: Check if the each element is a string, and support py2 as well.
        if not (isinstance(tiers, (tuple, list)) or callable(tiers)):
            raise TypeError(
                "`tiers` is expected to be a tuple or list of strings, or a predicate function, got: {}".
                format(tiers)
            )

        tiers = tuple(tiers) if not callable(tiers) else tiers

        warnemptytier = True
        if tiers == ():  # read all tiers
            tiers = tuple(eaf.get_tier_names())  # method returns dict_keys
            warnemptytier = False
        elif callable(tiers):
            tiers = tuple(name for name in eaf.get_tier_names() if tiers(name))

        if len(tiers) == 0:
            raise RuntimeError("No tiers found in the given file:\n{}".format(filepath))

        starts_ends = []
        labels = []
        samplerate = 1000  # NOTE: pympi only supports annotations in milliseconds

        for tier in tiers:
            annots = eaf.get_annotation_data_for_tier(tier)
            if warnemptytier and len(annots) == 0:
                warnings.warn(
                    RuntimeWarning(
                        "No annotations found for tier: {} in file\n{}.".format(
                            tier, filepath
                        )
                    )
                )
                continue

            n_rawannots = len(annots)
            annots = list(zip(*[a for a in annots if a[1] > a[0]]))
            # filter away annotations that are <= zero duration long

            if warnemptytier and len(annots[0]) < n_rawannots:
                warnings.warn(
                    RuntimeWarning(
                        "IGNORED {} zero- or negative-duration annotations of {} annotations in tier {} in file\n{}".
                        format(len(annots) - len(annots[0]), len(annots), tier, filepath)
                    )
                )

            starts_ends.extend(zip(*annots[:2]))
            attrs = eaf.tiers[tier][2]  # tier attributes
            contents = zip(*annots[2:])  # symbolic associations, etc.
            labels.extend(
                EAFAnnotationInfo(
                    tier,
                    annotator=attrs.get('ANNOTATOR', ""),
                    participant=attrs.get('PARTICIPANT', ""),
                    content=content,
                ) for content in contents
            )

        if len(starts_ends) == 0:
            raise RuntimeError(
                "All tiers {} were found to be empty in file\n{}".format(tiers, filepath)
            )

        if cls == SequenceLabels:
            return cls(starts_ends, labels, samplerate)
        else:
            # some child class
            # let's honor kwargs, they should too
            return starts_ends, labels, samplerate, kwargs

    def to_eaf(  # pylint: disable=too-many-arguments
            self,
            to_filepath=None,
            eafobj=None,
            linked_media_filepath=None,
            author="rennet.{}".format(rennet_version),
            annotinfo_fn=lambda label: EAFAnnotationInfo(tier_name=str(label)),
    ):
        labels = np.array(list(map(annotinfo_fn, self.labels)))
        assert all(
            isinstance(l, EAFAnnotationInfo) for l in labels
        ), "`annotinfo_fn` should return an `EafAnnotationInfo` object for each label"

        # flatten everything
        with self.samplerate_as(1000):  # pympi only supports milliseconds
            se, li = self._flattened_indices()
            if se.dtype != np.int:
                # EAF requires integers as starts and ends
                # IDEA: Warn rounding?
                se = np.rint(se).astype(np.int)  # pylint: disable=no-member

        if eafobj is None:
            eaf = Eaf(author=author)
            eaf.annotations = OrderedDict()
            eaf.tiers = OrderedDict()
            eaf.timeslots = OrderedDict()
        else:
            eaf = eafobj

        if linked_media_filepath is not None:
            try:
                eaf.add_linked_file(abspath(linked_media_filepath))
            except:  # pylint: disable=bare-except
                warnings.warn(
                    RuntimeWarning(
                        "Provided file was not added as linked file due to `pympi` errors. Provided File:\n{}\nError:\n{}".
                        format(linked_media_filepath, sys.exc_info())
                    )
                )

        # seen_tier_names = set()
        for (start, end), lix in zip(se, li):
            curr_seen_tier_names = set()
            if len(lix) > 0:
                for ann in labels[lix, ...]:
                    if ann.tier_name not in eaf.tiers:
                        # FIXME: handle different participant and annotator for same tier_name
                        eaf.add_tier(
                            ann.tier_name, part=ann.participant, ann=ann.annotator
                        )

                    if ann.tier_name in curr_seen_tier_names:
                        raise ValueError(
                            "Duplicate annotations on the same tier in "
                            "the same time-slot is not valid in ELAN.\n"
                            "Found at time-slot {} ms \n{}".format(
                                (start, end),
                                "\n".join(map(str, labels[lix, ...])),
                            )
                        )

                    eaf.add_annotation(
                        ann.tier_name,
                        start,
                        end,
                        value=ann.content,
                    )
                    curr_seen_tier_names.add(ann.tier_name)

        if to_filepath is not None:
            eaf.to_file(abspath(to_filepath))

        return eaf

    # TODO: [ ] Export to mpeg7

    # IDEA: [ ] Merge with other SequenceLabels, with label_fn to replace or overlap
    # IDEA: [ ] Extend other SequenceLabels, with label_fn to replace or overlap


class EAFAnnotationInfo(BaseSlotsOnlyClass):  # pylint: disable=too-few-public-methods
    """ Base individual annotation object from an ELAN file.
    Check `pympi` package for more information. ('pympi-ling' on pypi)
    """
    __slots__ = ("tier_name", "annotator", "participant", "content")

    def __init__(
        self,
        tier_name,
        annotator="rennet.{}".format(rennet_version),
        participant="",
        content=""
    ):
        self.tier_name = str(tier_name)
        self.annotator = str(annotator)
        self.participant = str(participant)
        self.content = str(content)


EafAnnotationInfo = EAFAnnotationInfo  # NOTE: for compatibility


class MPEG7AnnotationInfo(BaseSlotsOnlyClass):  # pylint: disable=too-few-public-methods
    """ Base individual annotation object from an MPEG7 file.
    Check `rennet.utils.mpeg7_utils` for more information.
    """
    __slots__ = ("speakerid", "gender", "givenname", "confidence", "content")

    def __init__(  # pylint: disable=too-many-arguments
            self,
            speakerid,
            gender="",
            givenname="",
            confidence="",
            content=""):
        self.speakerid = str(speakerid)
        self.gender = str(gender)
        self.givenname = str(givenname)
        self.confidence = str(confidence)
        self.content = str(content)


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

    def labels_at(self, ends, samplerate=None, default_label='zeros', rounded=10):
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
                dtype=self.labels.dtype,
            )
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
            labels, groupby_keyfn, keep, min_start, samplerate, **kwargs
        )
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

    @classmethod
    def from_mpeg7(cls, filepath, use_tags='ns', **kwargs):
        """ Create instance of ContiguousSequenceLabels from an mpeg7 annotation file.

        NOTE: Will raise error if the parsed annotations are not conitguous!!

        WARNING: Pretty hacked up solution. Check `rennet.utils.mpeg7_utils`.

        NOTE: if the callee `cls` is not ContiguousSequenceLabels, then no class is instantiated.
        It will be the responsibility of the callee (probably a child class) to create
        the appropriate instance of it's class.

        NOTE: Supported use_tags: "ns" (default), "mpeg7".
        """
        res = super(ContiguousSequenceLabels, cls).from_mpeg7(
            filepath, use_tags=use_tags, **kwargs
        )
        if cls == ContiguousSequenceLabels:
            return cls(*res)
        else:
            return res

    def calc_raw_viterbi_priors(
        self, state_keyfn=lambda label: label, samplerate=None, round_to_int=False
    ):
        """ Calculate raw priors for Markov states of labels. aka raw Viterbi priors.

        The Markov state corresponding to a label is determined by the calling the provided
        `state_keyfn`, which should return a value which can be collected into a Python Set.
        By default, a unit function is applied (i.e. each label is the state itself).

        Parameters
        ----------
        state_keyfn: function accepting one label and returning the hidden Markov state
            See Above.
        samplerate: int or float > 0, or None (default, check samplerate_as method)
            samplerate at which to calculate the priors.
            This is the safe parameter to change if you are getting priors that are floats
            Although, it does not guarantee that the results will be float, but,
            with an appropriately safe value chosen, you can force it to int later by passing `round_to_int` as True.
        round_to_int: bool, (default: False)
            Whether or not to round the priors calculations to integer values.
            This will be a hard and unsafe casting (yet following rounding rules), so
            only do this when it is safe to discard all values after the decimal point for each start/end.
            Try setting appropriate samplerate first to fix floating priors.

        Returns
        -------
        unique_states: array of unique hidden Markov states for the labels
        init: 1D numpy.ndarray of shape (len(unique_states), )
            with all zeros except for '1' at the index of the `unique_states`
            for the first label.
        trans: 2D numpy.ndarray of shape (len(unique_states), len(unique_states))
            Number of transitions (at the given `samplerate`) from one state to other.
        priors: 1D numpy.ndarray of shape (len(unique_states), )
            with number of occurrences (at the given `samplerate`) for each of the
            `unique_states`.
        """
        states = np.array(list(map(state_keyfn, self.labels)))
        unique_states = np.array(sorted(set(states)))

        state_ids = unique_states[np.newaxis, :] == states[..., np.newaxis]

        with self.samplerate_as(samplerate):
            durations = np.diff(self.starts_ends, axis=1)[..., 0]

            # FIXME: What should be done if durations is float with digits after decimal?
            # They won't make much difference since all the priors are going to get normalized later.
            # HACK: round the durations to int, and hence also all the priors later
            if round_to_int:
                durations = np.round(durations, 0).astype(np.int)
                # IDEA: At least warn for potential loss of information?

        init = state_ids[0, ...].astype(durations.dtype)
        priors = np.array(
            [durations[state_ids[:, s]].sum(axis=0) for s in range(len(unique_states))]
        ).astype(durations.dtype)
        confmatcat = confusion_matrix_forcategorical
        trans = confmatcat(state_ids[:-1, ...],
                           state_ids[1:, ...]).astype(durations.dtype)
        self_trans = (priors - state_ids.astype(np.int).sum(axis=0)
                      )  # e.g. segment of length 5 has 4 transitions to the same state

        # NOTE: Consecutive segments with the same state_id would
        # already have been accounted for in the confusion_matrix calculation
        trans.flat[::trans.shape[0] + 1] += self_trans

        return unique_states, init, trans, priors


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


# TODO: Funtion to extract viterbi priors from SequenceLabels


def normalize_raw_viterbi_priors(init, tran):
    assert init.shape[-1] == tran.shape[-1], "Shape mismatch between the inputs." +\
        " Both should be for the same number of classes (last dim)"
    return init / init.sum(), normalize_confusion_matrix(tran)[1]


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
