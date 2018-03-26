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
import warnings
from collections import namedtuple
import numpy as np
import h5py as h

from ..utils import label_utils as lu
from ..utils.py_utils import BaseSlotsOnlyClass
from ..utils import h5_utils as hu

samples_for_labelsat = lu.samples_for_labelsat  # pylint: disable=invalid-name
times_for_labelsat = lu.times_for_labelsat  # pylint: disable=invalid-name


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
        """Read mpeg7 annotations for KA3 data.

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

        # speakers = tuple(map(Speaker, *sorted(unique_speakers)))
        # Doesn't work, but the one below does, FML
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
        """Parse KA3 annotations from file.

        Parameters
        ----------
        filepath: path to a valid file
        use_tags: 'ns' or 'mpeg7' (optional, only when file is mpeg7 xml)
            Check `rennet.utils.mpeg7_utils`.
        tiers: list or tuple of strings or unary callable (optional, only when file is elan eaf)
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
        return (
            self.__class__(self.sourcefile, self.speakers, *args)
            if self.__class__ is Annotations else args
        )


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
                    "Some speakers may have duplicate annotations for file\n{}\nDUPLICATES IGNORED".
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
        return (
            self.__class__(self.sourcefile, self.speakers, *args)
            if self.__class__ is ActiveSpeakers else args
        )


# INPUTS PROVIDERS ################################################## INPUTS PROVIDERS #

Chunking = namedtuple(
    'Chunking', [
        'datapath',
        'dataslice',
        'swapchannels',
        'labelpath',
        'labelslice',
    ]
)


class H5ChunkingsReader(hu.BaseH5ChunkingsReader):
    def __init__(
            self,
            filepath,
            audios_root='audios',
            labels_root='labels',
            duplicate_swap_channels=True,
            **kwargs
    ):  # yapf: disable

        self.audios_root = audios_root
        self.labels_root = labels_root

        self._conversations = None
        self._dupswap_channels = duplicate_swap_channels

        self._totlen = None
        self._chunkings = None

        super(H5ChunkingsReader, self).__init__(filepath, **kwargs)

    def _read_all_conversations(self):
        conversations = tuple()

        # NOTE: Assuming labels and audios have the same group and dset names
        # use audio root to visit all the groups, and subsequent conversations
        # They group/
        with h.File(self.filepath, 'r') as f:
            root = f[self.audios_root]
            conversations += tuple(
                "{}/{}".format(group, conv)
                for group in root.keys()
                for conv in root[group].keys()
            )

        return conversations

    @property
    def conversations(self):
        if self._conversations is None:
            self._conversations = self._read_all_conversations()
        return self._conversations

    @conversations.setter
    def conversations(self, value):
        self._conversations = value

    @property
    def totlen(self):
        if self._totlen is None:
            self._read_all_chunkings()

        return self._totlen

    def _read_all_chunkings(self):
        # NOTE: We use the chunking info from the audios
        # and use the same for labels.
        chunkings = []
        total_len = 0

        with h.File(self.filepath, 'r') as f:
            audior = f[self.audios_root]
            labelr = f[self.labels_root]

            for conversation in sorted(self.conversations):
                audiod = audior[conversation]  # h5 Dataset
                labeld = labelr[conversation]  # h5 Dataset

                totlen = audiod.shape[0]

                chunksize = audiod.chunks[0] if audiod.chunks else totlen
                starts = np.arange(0, totlen, chunksize)
                ends = np.empty_like(starts)
                ends[-1] = totlen
                ends[:-1] = starts[1:]

                total_len += totlen
                chunkings.extend(
                    Chunking(
                        datapath=audiod.name,
                        dataslice=np.s_[s:e, ...],
                        swapchannels=False,
                        labelpath=labeld.name,
                        labelslice=np.s_[s:e, ...]
                    ) for s, e in zip(starts, ends)
                )

                if self._dupswap_channels:
                    if len(audiod.shape) < 3 or audiod.shape[-1] != 2:
                        msg = "Audio data does not seem to have 2 channels. "
                        msg += "Found chunk of shape: {}\n".format(audiod.shape)
                        msg += "Channels are expected to be the last dimension. "
                        msg += "Only stereo channels are supported."
                        raise ValueError(msg)

                    total_len += totlen
                    chunkings.extend(
                        Chunking(
                            datapath=audiod.name,
                            dataslice=np.s_[s:e, ...],
                            swapchannels=True,
                            labelpath=labeld.name,
                            labelslice=np.s_[s:e, ...]
                        ) for s, e in zip(starts, ends)
                    )

        self._totlen = total_len
        self._chunkings = chunkings

    @property
    def chunkings(self):
        if self._chunkings is None:
            self._read_all_chunkings()

        return self._chunkings

    @classmethod
    def for_conversations_at(  # pylint: disable=too-many-arguments
            cls,
            filepath,
            at=np.s_[:],
            audios_root='audios',
            labels_root='labels',
            duplicate_swap_channels=True,
            **kwargs
    ):  # yapf: disable
        obj = cls(
            filepath,
            audios_root=audios_root,
            labels_root=labels_root,
            duplicate_swap_channels=duplicate_swap_channels,
            **kwargs
        )

        if at == np.s_[:]:
            return obj

        allconversations = np.sort(obj.conversations)

        try:
            conversations_at = allconversations[at]
        except IndexError:
            print("\nTotal number of CallIDs: {}\n".format(len(allconversations)))
            raise

        if not isinstance(conversations_at, np.ndarray):
            # HACK: single conversation
            obj.conversations = (conversations_at, )
        else:
            obj.conversations = tuple(conversations_at)

        return obj


class CategoricalLabelsPrepper(hu.AsIsChunkPrepper):
    def prep_label(self, label, **kwargs):
        return label.astype(float)  # NOTE: Assuming stored labels are categorical

class FrameWithContextSubsamplingInputsProvider(  # pylint: disable=too-many-ancestors
        H5ChunkingsReader,
        CategoricalLabelsPrepper,
        hu.BaseWithContextClassSubsamplingSteppedInputsProvider,
):  # yapf: disable
    pass


class ChunkMeanVarianceNormalizingChannelSwappingCategoricalPrepper(
        hu.BaseChunkMeanVarianceNormalizer,
        CategoricalLabelsPrepper,
):  # yapf: disable
    def prep_data(self, data, only_labels=False, chunking=None, **kwargs):  # pylint: disable=arguments-differ
        if only_labels:  # This is dummy data, if only_labels
            return data

        if chunking is not None and chunking.swapchannels:
            data = data[..., ::-1]  # NOTE: Assuming last dim is for channels

        return self.normalize_data(data, **kwargs)


class ChMVNChannelSwappingFrameWithContextSubsamplingInputsProvider(  # pylint: disable=too-many-ancestors
        ChunkMeanVarianceNormalizingChannelSwappingCategoricalPrepper,
        FrameWithContextSubsamplingInputsProvider,
):  # yapf: disable
    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
            self,
            filepath,

            # audio ops
            audios_root='audios',
            data_context=0,
            duplicate_swap_channels=True,
            mean_it=False,
            std_it=False,
            add_channel_at_end=False,  # working with stereo

            # label ops
            labels_root='labels',
            label_subcontext=0,  # 0 for choosing only the center label as label
            label_from_subcontext_fn=hu.dominant_label_for_subcontext,
            nclasses=3,

            # sub-sampling ops
            classkeyfn=np.argmax,  # for categorical labels
            class_subsample_to_ratios=1.,  # float, tuple or dict, default keeps all

            # flow ops
            npasses=1,
            steps_per_chunk=1,
            shuffle_seed=None,
            **kwargs):

        # Mainly here for documentation and auto-completion
        sup = super(ChMVNChannelSwappingFrameWithContextSubsamplingInputsProvider, self)
        sup.__init__(
            filepath,
            audios_root=audios_root,
            labels_root=labels_root,
            data_context=data_context,
            label_subcontext=label_subcontext,
            label_from_subcontext_fn=label_from_subcontext_fn,
            classkeyfn=classkeyfn,
            class_subsample_to_ratios=class_subsample_to_ratios,
            steps_per_chunk=steps_per_chunk,
            shuffle_seed=shuffle_seed,
            npasses=npasses,
            add_channel_at_end=add_channel_at_end,
            duplicate_swap_channels=duplicate_swap_channels,
            mean_it=mean_it,
            std_it=std_it,
            nclasses=nclasses,
            **kwargs
        )
