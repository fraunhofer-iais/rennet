"""
@motjuste
Created: 29-08-2016

Helpers for working with KA3 dataset
"""
from __future__ import print_function, division
import xml.etree.ElementTree as et
import numpy as np
import warnings
from functools import reduce

import rennet.utils.label_utils as lu
from rennet.utils.py_utils import BaseSlotsOnlyClass, lowest_common_multiple

samples_for_labelsat = lu.samples_for_labelsat
times_for_labelsat = lu.times_for_labelsat

MPEG7_NAMESPACES = {
    "ns": "http://www.iais.fraunhofer.de/ifinder",
    "ns2": "urn:mpeg:mpeg7:schema:2004",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "mpeg7": "urn:mpeg:mpeg7:schema:2004",
    "ifinder": "http://www.iais.fraunhofer.de/ifinder"
}

NS2_TAGS = {
    "audiosegment": ".//ns2:AudioSegment",
    "timepoint": ".//ns2:MediaTimePoint",
    "duration": ".//ns2:MediaDuration",
    "descriptor": ".//ns2:AudioDescriptor[@xsi:type='SpokenContentType']",
    "speakerid": ".//ns:Identifier",
    "transcription": ".//ns:SpokenUnitVector",
    "confidence": ".//ns:ConfidenceVector",
    "speakerinfo": ".//ns:Speaker",
    "gender": "gender",
    "givenname": ".//ns2:GivenName",
}

MPEG7_TAGS = {
    "audiosegment":
    ".//mpeg7:AudioSegment",
    "timepoint":
    ".//mpeg7:MediaTimePoint",
    "duration":
    ".//mpeg7:MediaDuration",
    "descriptor":
    ".//mpeg7:AudioDescriptor[@xsi:type='ifinder:SpokenContentType']",
    "speakerid":
    ".//ifinder:Identifier",
    "transcription":
    ".//ifinder:SpokenUnitVector",
    "confidence":
    ".//ifinder:ConfidenceVector",
    "speakerinfo":
    ".//ifinder:Speaker",
    "gender":
    "gender",
    "givenname":
    ".//mpeg7:GivenName",
}


def _parse_timepoint(timepoint):
    _, timepoint = timepoint.split('T')  # 'T' indicates the rest part is time
    hours, minutes, sec, timepoint = timepoint.split(':')

    # timepoint will have the nFN
    # n = number of fraction of seconds
    # N = the standard number of fractions per second
    val, persec = timepoint.split('F')

    res = int(hours) * 3600 +\
          int(minutes) * 60 +\
          int(sec)

    return res * int(persec) + int(val), int(persec)


def _parse_duration(duration):
    _, duration = duration.split('T')

    splits = [0] * 5
    for i, marker in enumerate(['H', 'M', 'S', 'N', 'F']):
        if marker in duration:
            value, duration = duration.split(marker)
            splits[i] = int(value)
        else:
            splits[i] = 0

    hours, minutes, sec, val, persec = splits
    res = int(hours) * 3600 +\
          int(minutes) * 60 +\
          int(sec)

    return res * int(persec) + int(val), int(
        persec)  # need to send separately as the float sum is not great


def _parse_timestring(timepoint, duration):
    tpt, tps = _parse_timepoint(timepoint)
    dur, dps = _parse_duration(duration)

    persec = lowest_common_multiple(tps, dps)
    tpt *= (persec // tps)
    return tpt, tpt + dur * (persec // dps), persec


def _parse_segment(segment, TAGS):
    timepoint = segment.find(TAGS["timepoint"], MPEG7_NAMESPACES).text
    duration = segment.find(TAGS["duration"], MPEG7_NAMESPACES).text
    descriptor = segment.find(TAGS["descriptor"], MPEG7_NAMESPACES)

    if any(d is None for d in [timepoint, duration]):  #, descriptor]):
        raise ValueError(
            "timepoint, duration or decriptor not found in segment")

    start_end_persec = _parse_timestring(timepoint, duration)

    return start_end_persec, descriptor


def _parse_descriptor(descriptor, TAGS):
    speakerid = descriptor.find(TAGS["speakerid"], MPEG7_NAMESPACES).text
    speakerinfo = descriptor.find(TAGS["speakerinfo"], MPEG7_NAMESPACES)
    transcription = descriptor.find(TAGS["transcription"],
                                    MPEG7_NAMESPACES).text
    confidence = descriptor.find(TAGS["confidence"], MPEG7_NAMESPACES).text

    gender = speakerinfo.get(TAGS["gender"])
    givenname = speakerinfo.find(TAGS["givenname"], MPEG7_NAMESPACES).text

    if any(x is None
           for x in [speakerid, gender, givenname, confidence, transcription]):
        raise ValueError("Some descriptor information is None / not found")

    return speakerid, gender, givenname, confidence, transcription


def _sanitize_starts_ends(starts_ends, persecs):
    """ Sanitize starts ends to be of the same samplerate (persec) """
    persec = reduce(lowest_common_multiple, set(persecs))
    return [(s * persec // p, e * persec // p)
            for (s, e), p in zip(starts_ends, persecs)], persec


def parse_mpeg7(filepath, use_tags="ns"):  # pylint: disable=too-many-locals
    """ Parse MPEG7 speech annotations into lists of data

    """
    tree = et.parse(filepath)
    root = tree.getroot()

    if use_tags == "mpeg7":
        TAGS = MPEG7_TAGS
    elif use_tags == "ns":
        TAGS = NS2_TAGS

    # find all AudioSegments
    segments = root.findall(TAGS["audiosegment"], MPEG7_NAMESPACES)
    if len(segments) == 0:
        raise ValueError("No AudioSegment tags found")

    starts_ends = []
    persecs = []
    speakerids = []
    genders = []
    givennames = []
    confidences = []
    transcriptions = []
    for i, s in enumerate(segments):
        try:
            start_end_persec, descriptor = _parse_segment(s, TAGS)
        except ValueError:
            print("Segment number :%d" % (i + 1))
            raise

        if descriptor is None:
            # if there is not descriptor, there is no speech. Ignore!
            continue

        if start_end_persec[1] <= start_end_persec[0]:  # (end - start) <= 0
            warnings.warn(
                "(end - start) <= 0 ignored for annotation at {} with values {} in file {}".
                format(i, start_end_persec, filepath))
            continue

        starts_ends.append(start_end_persec[:-1])
        persecs.append(start_end_persec[-1])

        try:
            si, g, gn, conf, tr = _parse_descriptor(descriptor, TAGS)
        except ValueError:
            print("Segment number:%d" % (i + 1))

        speakerids.append(si)
        genders.append(g)
        givennames.append(gn)
        confidences.append(conf)
        transcriptions.append(tr)

    starts_ends, persecs = _sanitize_starts_ends(starts_ends, persecs)

    return (starts_ends, persecs, speakerids, genders, givennames, confidences,
            transcriptions)


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
    def from_file(cls, filepath, use_tags="ns"):  # pylint: disable=too-many-locals
        se, sr, sids, gen, gn, conf, trn = parse_mpeg7(filepath, use_tags)

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
                "No Annotations were found from file {}. Check use_tags parameter.".
                format(filepath))

        return cls(
            filepath,
            tuple(speakers),
            starts_ends,
            transcriptions,
            samplerate=sr, )

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
                labels[i] = (
                    spks == ann.labels[lix[0]].speakerid).astype(np.int)
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
            samplerate=ann.samplerate, )

    @classmethod
    def from_file(cls, filepath, warn_duplicates=True, use_tags="ns"):
        ann = Annotations.from_file(filepath, use_tags)
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
