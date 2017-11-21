"""
@motjuste
Created: 29-08-2016

Helpers for working with KA3 dataset
"""
from __future__ import print_function, division
import numpy as np
import warnings

import rennet.utils.label_utils as lu
from rennet.utils.py_utils import BaseSlotsOnlyClass
from rennet.utils.mpeg7_utils import parse_mpeg7

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
    def from_file(cls, filepath, **kwargs):  # pylint: disable=too-many-locals
        se, sr, sids, gen, gn, conf, trn = parse_mpeg7(filepath, **kwargs)

        uniq_sids = sorted(set(sids))

        speakers = []
        for sid in uniq_sids:
            i = sids.index(sid)
            speakers.append(Speaker(sid, gen[i], gn[i]))

        starts_ends = []
        transcriptions = []
        for i, (s, e) in enumerate(se):
            starts_ends.append((s, e))
            transcriptions.append(
                Transcription(sids[i], float(conf[i]), trn[i]))

        if len(starts_ends) == 0:
            raise RuntimeError(
                "No Annotations were found from file {}.\n".format(filepath) + \
                "Check `use_tags` parameter for `mpeg7_utils.parse_mpeg7` "+\
                "and pass appropriate one as keyword argument to this function.\n"+\
                "Options: 'ns' (default) and 'mpeg7'"
                )

        return cls(
            filepath,
            tuple(speakers),
            starts_ends,
            transcriptions,
            samplerate=sr,
        )

    def idx_for_speaker(self, speaker):
        speakerid = speaker.speakerid
        for i, l in enumerate(self.labels):
            if l.speakerid == speakerid:
                yield i

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
        super().__init__(starts_ends, labels, samplerate)

    @classmethod
    def from_annotations(cls, ann, warn_duplicates=True):
        starts_ends, labels_idx = ann._flattened_indices()  # pylint: disable=protected-access

        spks = np.array([s.speakerid for s in ann.speakers])
        labels = np.zeros(shape=(len(starts_ends), len(spks)), dtype=np.int)
        for i, lix in enumerate(labels_idx):
            if len(lix) == 1:
                labels[i] = (spks == ann.labels[lix[0]].speakerid).astype(
                    np.int)
            elif len(lix) > 1:
                lspks = np.array([ann.labels[s].speakerid for s in lix])
                labels[i] = (spks == lspks[:, None]).sum(axis=0)

        if labels.max() > 1:
            labels[labels > 1] = 1
            if warn_duplicates:
                warnings.warn(
                    "Some speakers may have duplicate annotations for file {}.\nDUPLICATES IGNORED".
                    format(ann.sourcefile))

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
