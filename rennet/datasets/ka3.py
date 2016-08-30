"""
@motjuste
Created: 29-08-2016

Helpers for working with KA3 dataset
"""
from __future__ import print_function
from collections import namedtuple
import xml.etree.ElementTree as et
import numpy as np

import rennet.utils.label_utils as lu

MPEG7_NAMESPACES = {
    "ns": "http://www.iais.fraunhofer.de/ifinder",
    "ns2": "urn:mpeg:mpeg7:schema:2004",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}

MPEG7_TAGS = {
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


def _parse_timepoint(timepoint):
    _, timepoint = timepoint.split('T')  # 'T' indicates the rest part is time
    hours, minutes, sec, timepoint = timepoint.split(':')

    # timepoint will have the nFN
    # n = number of fraction of seconds
    # N = the standard number of fractions per second
    val, persec = timepoint.split('F')

    res = (int(hours) * 3600 + int(minutes) * 60 + int(sec) + int(val) /
           int(persec))

    return res


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
    res = (int(hours) * 3600 + int(minutes) * 60 + int(sec) + int(val) /
           int(persec))

    return res


def _parse_timestring(timepoint, duration):
    tp = _parse_timepoint(timepoint)
    dur = _parse_duration(duration)

    return tp, tp + dur


def _parse_segment(segment):
    timepoint = segment.find(MPEG7_TAGS["timepoint"], MPEG7_NAMESPACES).text
    duration = segment.find(MPEG7_TAGS["duration"], MPEG7_NAMESPACES).text
    descriptor = segment.find(MPEG7_TAGS["descriptor"], MPEG7_NAMESPACES)

    if any(d is None for d in [timepoint, duration, descriptor]):
        raise ValueError(
            "timepoint, duration or decriptor not found in segment")

    start_end = _parse_timestring(timepoint, duration)

    return start_end, descriptor


def _parse_descriptor(descriptor):
    speakerid = descriptor.find(MPEG7_TAGS["speakerid"], MPEG7_NAMESPACES).text
    speakerinfo = descriptor.find(MPEG7_TAGS["speakerinfo"], MPEG7_NAMESPACES)
    transcription = descriptor.find(MPEG7_TAGS["transcription"],
                                    MPEG7_NAMESPACES).text
    confidence = descriptor.find(MPEG7_TAGS["confidence"],
                                 MPEG7_NAMESPACES).text

    gender = speakerinfo.get(MPEG7_TAGS["gender"])
    givenname = speakerinfo.find(MPEG7_TAGS["givenname"],
                                 MPEG7_NAMESPACES).text

    if any(x is None
           for x in [speakerid, gender, givenname, confidence, transcription]):
        raise ValueError("Some descriptor information is None / not found")

    return speakerid, gender, givenname, confidence, transcription


# pylint: disable=too-many-locals
def parse_mpeg7(filepath):
    """ Parse MPEG7 speech annotations into lists of data

    """
    tree = et.parse(filepath)
    root = tree.getroot()

    # find all AudioSegments
    segments = root.findall(MPEG7_TAGS["audiosegment"], MPEG7_NAMESPACES)
    if len(segments) == 0:
        raise ValueError("No AudioSegment tags found")

    starts_ends = []
    speakerids = []
    genders = []
    givennames = []
    confidences = []
    transcriptions = []
    for i, s in enumerate(segments):
        try:
            startend, descriptor = _parse_segment(s)
        except ValueError:
            print("Segment number :%d" % (i + 1))
            raise

        if startend[1] <= startend[0]:  # (end - start) <= 0
            continue

        starts_ends.append(startend)

        try:
            si, g, gn, conf, tr = _parse_descriptor(descriptor)
        except ValueError:
            print("Segment number:%d" % (i + 1))

        speakerids.append(si)
        genders.append(g)
        givennames.append(gn)
        confidences.append(conf)
        transcriptions.append(tr)

    return (starts_ends, speakerids, genders, givennames, confidences,
            transcriptions)
# pylint: enable=too-many-locals

Speaker = namedtuple('Speaker', ['speakerid', 'gender', 'givenname'])

Transcription = namedtuple('Transcription', [
    'speakerid', 'confidence', 'content'
])


class Annotations(lu.SequenceLabels):
    def __init__(self, filepath, speakers, *args, **kwargs):
        self.sourcefile = filepath
        self.speakers = speakers
        super().__init__(*args, **kwargs)

    # pylint: disable=too-many-locals
    @classmethod
    def from_file(cls, filepath):
        se, sids, gen, gn, conf, trn = parse_mpeg7(filepath)

        uniq_sids = sorted(set(sids))

        speakers = []
        for sid in uniq_sids:
            i = sids.index(sid)
            speakers.append(Speaker(sid, gen[i], gn[i]))

        starts_ends = []
        transcriptions = []
        for i, (s, e) in enumerate(se):
            starts_ends.append((s, e))
            transcriptions.append(Transcription(sids[i], float(conf[i]), trn[
                i]))

        return cls(filepath,
                   speakers,
                   starts_ends,
                   transcriptions,
                   samplerate=1)
    # pylint: enable=too-many-locals

    def idx_for_speaker(self, speaker):
        speakerid = speaker.speakerid
        for i, l in enumerate(self.labels):
            if l.speakerid == speakerid:
                yield i


class ActiveSpeakers(Annotations):
    def __init__(self, filepath, speakers, *args, **kwargs):
        super().__init__(filepath, speakers, *args, **kwargs)

    @staticmethod
    def group_by_values(values):
        # TODO: [A] add to a generic util module
        # TODO: [A] make it work with 1 dimensional arrays
        # Ref: http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance

        initones = [[1] * values.shape[1]]
        diff = np.concatenate([initones, np.diff(values, axis=0)])
        starts = np.unique(np.where(diff)[0])  # remove duplicate starts

        ends = np.ones(len(starts), dtype=np.int)
        ends[:-1] = starts[1:]
        ends[-1] = len(values)

        labels = values[starts]

        return np.vstack([starts, ends]).T, labels


    @classmethod
    def from_annotations(cls, ann, samplerate=100):  # default 100 for ka3
        with ann.samplerate_as(samplerate):
            se_ = ann.starts_ends
            se = se_.astype(np.int)

            # TODO: [A] better error statement
            # FIXME: [A] the below assert fails for something that I think is correct
            # assert all(se == se_), "The provided sample rate does not cover all decimals"

        n_speakers = len(ann.speakers)
        total_duration = se[:, 1].max()
        active_speakers = np.zeros(shape=(total_duration, n_speakers), dtype=np.int)

        for s, speaker in enumerate(ann.speakers):
            for i in ann.idx_for_speaker(speaker):
                start, end = se[i]
                active_speakers[start:end, s] += 1

        starts_ends, active_speakers = cls.group_by_values(active_speakers)

        return cls(
            ann.sourcefile, ann.speakers,
            starts_ends, active_speakers, samplerate=samplerate
        )

    @classmethod
    def from_file(cls, filepath):
        return cls.from_annotations(super().from_file(filepath))
