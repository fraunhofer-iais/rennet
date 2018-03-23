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
"""Helpers for working with KA3 dataset

@motjuste
Created: 29-08-2016
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import warnings

from ..utils import label_utils as lu
from ..utils.py_utils import BaseSlotsOnlyClass

samples_for_labelsat = lu.samples_for_labelsat
times_for_labelsat = lu.times_for_labelsat


class Speaker(BaseSlotsOnlyClass):  # pylint: disable=too-few-public-methods
    __slots__ = ('speakerid', 'gender', 'givenname')

    def __init__(self, speakerid, gender, givenname):
        self.speakerid = speakerid
        self.gender = gender
        self.givenname = givenname


class Transcription(BaseSlotsOnlyClass):  # pylint: disable=too-few-public-methods
    __slots__ = ('speakerid', 'confidence', 'content')

    def __init__(self, speakerid, confidence, content):
        self.speakerid = speakerid
        self.confidence = confidence
        self.content = content


class Annotations(lu.SequenceLabels):
    # PARENT'S SLOTS
    # __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate',
    #              '_minstart_at_orig_sr', )
    __slots__ = ('sourcefile', 'speakers')

    def __init__(self, filepath, speakers, starts_ends, labels, samplerate=1):  # pylint: disable=too-many-arguments
        self.sourcefile = filepath
        self.speakers = speakers
        super(Annotations, self).__init__(starts_ends, labels, samplerate)

    @classmethod
    def from_mpeg7(cls, filepath, use_tags='ns', **kwargs):
        """ Read mpeg7 annotations for KA3 data.

        NOTE: Supported use_tags: "ns" (default), "mpeg7".
        Check `rennet.utils.mpeg7_utils`.
        """
        starts_ends, _labels, sr, _ = super(Annotations, cls).from_mpeg7(
            filepath, use_tags=use_tags, **kwargs
        )

        unique_speakers = set()
        labels = []
        for l in _labels:  # _labels is list of lu.MPEG7AnnotationInfo
            unique_speakers.add((l.speakerid, l.gender, l.givenname))

            # FIXME: only keeping speakerid for Transcription will confuse when
            # there are mutliple speakers with the same speakerid
            labels.append(Transcription(l.speakerid, float(l.confidence), l.content))

        # speakers = tuple(map(Speaker, *sorted(unique_speakers)))  # Doesn't work, but the one below does, FML
        speakers = tuple(Speaker(*uspk) for uspk in sorted(unique_speakers))
        return cls(filepath, speakers, starts_ends, labels, samplerate=sr)

    @classmethod
    def from_eaf(cls, filepath, tiers=(), **kwargs):
        parsed = super(Annotations, cls).from_eaf(filepath, tiers, **kwargs)
        starts_ends, _labels, samplerate, _ = parsed

        unique_speakers = set()
        labels = []
        for l in _labels:
            unique_speakers.add((l.tier_name, None,
                                 l.participant))  # Gender hasn't been deciphered
            labels.append(
                Transcription(speakerid=l.tier_name, confidence=1.0, content=l.content)
            )

        speakers = tuple(Speaker(*uspk) for uspk in sorted(unique_speakers))

        return cls(filepath, speakers, starts_ends, labels, samplerate=samplerate)

    @classmethod
    def from_file(cls, filepath, **kwargs):  # pylint: disable=too-many-locals
        """ Parse KA3 annotations from file.

        Parameters
        ----------
        filepath: path to a valid file
        use_tags: 'ns' or 'mpeg7' (optional, valid only when file is mpeg7 xml)
            Check `rennet.utils.mpeg7_utils`.
        tiers: list or tuple of strings or unary callable (optional, valid only when file is elan eaf)
            list or tuple of tier names (as strings) to specify what tiers to be read.
            By default, this is an empty tuple (or list), and all tiers will be read.
            If it is an unary callable (i.e. taking one argument), the function will be
            used as the predicated to filter which tiers should be kept.
        """
        # We have dug ourselves in a hole here by using a generic name and
        # by being able to support reading ELAN files

        # HACK: Relying on filename extensions to decide between elan eaf and mpeg7 xml
        parser = cls.from_eaf if filepath.lower().endswith(".eaf") else cls.from_mpeg7
        return parser(filepath, **kwargs)

    def __str__(self):
        s = "Source filepath: {}".format(self.sourcefile)
        s += "\nSpeakers: {}\n".format(len(self.speakers))
        s += "\n".join(str(s) for s in self.speakers)
        s += "\n" + super(Annotations, self).__str__()
        return s

    def __getitem__(self, idx):
        args = super(Annotations, self).__getitem__(idx)
        if self.__class__ is Annotations:
            return self.__class__(self.sourcefile, self.speakers, *args)
        else:
            return args


class ActiveSpeakers(lu.ContiguousSequenceLabels):
    # PARENT'S SLOTS
    # __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate',
    #              '_minstart_at_orig_sr', )
    __slots__ = ('sourcefile', 'speakers')

    def __init__(self, filepath, speakers, starts_ends, labels, samplerate=1):  # pylint: disable=too-many-arguments
        self.sourcefile = filepath
        self.speakers = speakers
        super(ActiveSpeakers, self).__init__(starts_ends, labels, samplerate)

    @classmethod
    def from_annotations(cls, ann, warn_duplicates=True):
        starts_ends, labels_idx = ann._flattened_indices()  # pylint: disable=protected-access

        spks = np.array([s.speakerid for s in ann.speakers])
        labels = np.zeros(shape=(len(starts_ends), len(spks)), dtype=np.int)
        for i, lix in enumerate(labels_idx):
            if len(lix) == 1:
                labels[i] = (spks == ann.labels[lix[0]].speakerid).astype(np.int)
            elif len(lix) > 1:
                lspks = np.array([ann.labels[s].speakerid for s in lix])
                labels[i] = (spks == lspks[:, None]).sum(axis=0)

        if labels.max() > 1:
            labels[labels > 1] = 1
            if warn_duplicates:
                warnings.warn(
                    "Some speakers may have duplicate annotations for file {}.\nDUPLICATES IGNORED".
                    format(ann.sourcefile)
                )

        # IDEA: merge consecutive segments with the same label
        # No practical impact expected, except probably in turn-taking calculations
        # It may result in extra continuations than accurate
        # Let's keep in mind though that this is being inferred from the annots,
        # i.e. these zero-gap continuations were separately annotated, manually,
        # and we should respect their segmentation
        # The turn-taking calculator should keep this in mind

        return cls(
            ann.sourcefile,
            ann.speakers,
            starts_ends,
            labels,
            samplerate=ann.samplerate,
        )

    @classmethod
    def from_file(cls, filepath, warn_duplicates=True, **kwargs):
        ann = Annotations.from_file(filepath, **kwargs)
        return cls.from_annotations(ann, warn_duplicates)

    def __str__(self):
        s = "Source filepath: {}".format(self.sourcefile)
        s += "\nSpeakers: {}\n".format(len(self.speakers))
        s += "\n".join(str(s) for s in self.speakers)
        s += "\n" + super(ActiveSpeakers, self).__str__()
        return s

    def __getitem__(self, idx):
        args = super(ActiveSpeakers, self).__getitem__(idx)
        if self.__class__ is ActiveSpeakers:
            return self.__class__(self.sourcefile, self.speakers, *args)
        else:
            return args
