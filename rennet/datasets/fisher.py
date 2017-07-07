"""
@motjuste
Created: 01-02-2017

Helpers for working with Fisher dataset
"""
from __future__ import print_function, division
import numpy as np
import warnings
from os.path import abspath
from csv import reader
import h5py as h

import rennet.utils.label_utils as lu
from rennet.utils.py_utils import BaseSlotsOnlyClass
import rennet.utils.np_utils as nu
import rennet.utils.training_utils as tu

samples_for_labelsat = lu.samples_for_labelsat
times_for_labelsat = lu.times_for_labelsat


class Speaker(BaseSlotsOnlyClass):  # pylint: disable=too-few-public-methods
    __slots__ = ('pin', 'gender', 'dialect')

    def __init__(self, pin, gender, dialect):
        self.pin = pin
        self.gender = gender
        self.dialect = dialect


class CallData(BaseSlotsOnlyClass):  # pylint: disable=too-few-public-methods
    __slots__ = ('callid', 'topicid', 'signalgrade', 'convgrade',
                 'channelspeakers')

    def __init__(  # pylint: disable=too-many-arguments
            self, callid, topicid, signalgrade, convgrade, channelspeakers):
        self.callid = callid
        self.topicid = topicid
        self.signalgrade = signalgrade
        self.convgrade = convgrade
        self.channelspeakers = channelspeakers  # List, in order [A, B]

    def __repr__(self):
        r = super(CallData, self).__repr__()
        # Make the list of Channel Speakers look good, maybe ... subjective
        r = r.replace('[', '\n\t\t[').replace('), ', '), \n\t\t ')
        return r


callid_for_filename = lambda fn: fn.split('_')[-1].split('.')[0]

groupid_for_callid = lambda callid: callid[:3]


class AllCallData(object):
    def __init__(self, allcalldata):
        self.allcalldata = sorted(allcalldata, key=lambda c: c.callid)
        self._callid_idx = {
            callid: i
            for i, callid in enumerate(c.callid for c in self.allcalldata)
        }

    def for_callid(self, callid):
        return self[callid]

    def for_filename(self, filename):
        return self[filename]

    @classmethod
    def from_file(cls, filepath):  # pylint: disable=too-many-locals
        afp = abspath(filepath)

        allcalldata = []
        with open(afp, 'r') as f:
            rdr = reader(f, delimiter=',')

            for i, row in enumerate(rdr):
                if i == 0:
                    # Header
                    continue

                callid, _, topicid, siggrade, convgrade = row[:5]
                apin, agendia, _, _, _ = row[5:-5]
                bpin, bgendia, _, _, _ = row[-5:]

                agen, adia = agendia.split('.')
                bgen, bdia = bgendia.split('.')

                calldata = CallData(
                    callid,
                    topicid,
                    float(siggrade),
                    float(convgrade),
                    [Speaker(apin, agen, adia), Speaker(bpin, bgen, bdia)], )
                allcalldata.append(calldata)

        return cls(allcalldata)

    def __getitem__(self, idx_or_callid_or_filename):
        idx = idx_or_callid_or_filename
        if isinstance(idx, str):  # callid or filename
            if len(idx) > 5:  # not a Fisher Callid
                # NOTE: Assuming, probably naively, that this a Fisher filename
                idx = callid_for_filename(idx)
            return self.allcalldata[self._callid_idx[idx]]
        elif isinstance(idx, (slice, int)):
            return self.allcalldata[idx]
        elif any(isinstance(i, str) for i in idx):
            return [self[i] for i in idx]
        else:
            raise TypeError('Unsupported index {} for {}'.format(
                idx, self.__class__.__name__))


class Transcription(BaseSlotsOnlyClass):  # pylint: disable=too-few-public-methods
    __slots__ = ('speakerchannel', 'content')

    def __init__(self, speakerchannel, content):
        self.speakerchannel = speakerchannel  # 0 for A, 1 for B
        self.content = content


class Annotations(lu.SequenceLabels):
    """
    TODO: [ ] Add proper docs

    NOTE: This is almost identical to ka3.Annotations, but copied here, cuz
    - Fisher is the main dataset for me.
    - ka3 module was not designed to be sub-classed.
    - Don't want headache of maintaining compatibility where I don't have to
    """

    # PARENT'S SLOTS
    # __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate')
    __slots__ = ('sourcefile', 'calldata')

    def __init__(self, filepath, calldata, *args, **kwargs):
        self.sourcefile = filepath
        self.calldata = calldata

        super(Annotations, self).__init__(*args, **kwargs)

    @property
    def callid(self):
        if self.calldata is None:
            # filenames are fe_03_CALLID.*
            return callid_for_filename(self.sourcefile)
        else:
            return self.calldata.callid

    def find_and_set_calldata(self, allcalldata):
        try:
            if isinstance(allcalldata, str):  # filename, probably
                allcalldata = AllCallData.from_file(allcalldata)

            self.calldata = allcalldata[self.sourcefile]
        except KeyError:
            raise KeyError(
                "Call Data for callid {} not found in provided allcalldata".
                format(self.callid))

    @classmethod
    def from_file(cls, filepath, allcalldata=None, skip_content=False):
        afp = abspath(filepath)

        se = []
        trans = []

        with open(afp, 'r') as f:
            rdr = reader(f, delimiter=':')

            for row in rdr:
                if len(row) == 0 or row[0][0] == '#':
                    # ignore empty lines or comments
                    continue
                else:
                    s, e, spk = row[0].split(' ')
                    se.append((float(s), float(e)))

                    spk = spk.strip()
                    if skip_content:
                        content = ''
                    else:
                        content = row[1].strip()
                    if spk.upper() == 'A':
                        trans.append(Transcription(0, content))
                    elif spk.upper() == 'B':
                        trans.append(Transcription(1, content))
                    else:
                        raise ValueError(
                            "Speaker channel other than A and B ({}) in file\n{}".
                            format(spk, filepath))

        if allcalldata is None:
            calldata = None
        elif isinstance(allcalldata, str):  # filepath, probably
            calldata = AllCallData.from_file(allcalldata)[afp]
        else:
            calldata = allcalldata[afp]  # AllCallData can accept filename too!

        return cls(afp, calldata, se, trans, samplerate=1)

    def __str__(self):
        s = "Source filepath:\n{}\n".format(self.sourcefile)
        s += "\nCalldata:\n{}\n".format(self.calldata)
        s += "\n" + super(Annotations, self).__str__()
        return s

    def __getitem__(self, idx):
        args = super(Annotations, self).__getitem__(idx)
        if self.__class__ is Annotations:
            return self.__class__(self.sourcefile, self.calldata, *args)
        else:
            return args


class ActiveSpeakers(lu.ContiguousSequenceLabels):
    # PARENT'S SLOTS
    # __slots__ = ('_starts_ends', 'labels', '_orig_samplerate', '_samplerate')
    __slots__ = ('sourcefile', 'calldata')

    def __init__(self, filepath, calldata, *args, **kwargs):
        self.sourcefile = filepath
        self.calldata = calldata

        super(ActiveSpeakers, self).__init__(*args, **kwargs)

        # SequenceLabels makes labels into a list
        self.labels = np.array(self.labels)

    @property
    def callid(self):
        if self.calldata is None:
            # filenames are fe_03_CALLID.*
            return callid_for_filename(self.sourcefile)
        else:
            return self.calldata.callid

    def find_and_set_calldata(self, allcalldata):
        try:
            if isinstance(allcalldata, str):  # filename, probably
                allcalldata = AllCallData.from_file(allcalldata)

            self.calldata = allcalldata[self.sourcefile]
        except KeyError:
            raise KeyError(
                "Call Data for callid {} not found in provided allcalldata".
                format(self.callid))

    @classmethod
    def from_annotations(cls, ann, samplerate=100,
                         warn=True):  # min time resolution 1ms, mostly
        """
        TODO: [ ] Better handling of warnings?
            The user should be aware that there is a problem,
            and some implicit decisions were made
            Hence `warn = True` by default
        """
        with ann.samplerate_as(samplerate):
            _se = ann.starts_ends
            se = np.round(_se).astype(np.int)

        if warn:
            try:
                np.testing.assert_almost_equal(se, _se)
            except AssertionError:
                _w = "Sample rate {} does not evenly divide all the starts and ends for file:\n{}".format(
                    samplerate, ann.sourcefile)
                warnings.warn(_w)

        # make contigious array of shape (total_duration, n_speakers)
        # NOTE: n_speakers is 2 for all Fisher data
        n_speakers = 2
        active_speakers = np.zeros(
            shape=(se[:, 1].max(), n_speakers), dtype=np.int)

        for (start, end), l in zip(se, ann.labels):
            # NOTE: not setting to 1 straightaway to catch duplicates
            active_speakers[start:end, l.speakerchannel] += 1

        if active_speakers.max() > 1:
            if warn:
                _w = "Some speakers may have duplicate annotations for file:\n{}.\n!!! IGNORED !!!".format(
                    ann.sourcefile)
                warnings.warn(_w)

            active_speakers[active_speakers > 1] = 1

        starts_ends, active_speakers = nu.group_by_values(active_speakers)

        return cls(
            ann.sourcefile,
            ann.calldata,
            starts_ends,
            active_speakers,
            samplerate=samplerate)

    @classmethod
    def from_file(cls, filepath, samplerate=100, allcalldata=None, warn=True):
        ann = Annotations.from_file(filepath, allcalldata, skip_content=True)
        # min time resolution 1ms, mostly
        return cls.from_annotations(ann, samplerate=samplerate, warn=warn)

    def __str__(self):
        s = "Source filepath:\n{}\n".format(self.sourcefile)
        s += "\nCalldata:\n{}\n".format(self.calldata)
        s += "\n" + super(ActiveSpeakers, self).__str__()
        return s

    def __getitem__(self, idx):
        args = super(ActiveSpeakers, self).__getitem__(idx)
        if self.__class__ is ActiveSpeakers:
            return self.__class__(self.sourcefile, self.calldata, *args)
        else:
            return args


class FisherH5ChunkingsReader(tu.BaseH5ChunkingsReader):
    def __init__(self,
                 filepath,
                 audios_root='audios',
                 labels_root='labels',
                 **kwargs):

        super(FisherH5ChunkingsReader, self).__init__(filepath, **kwargs)

        self.audios_root = audios_root
        self.labels_root = labels_root

        self.grouped_callids = self._read_all_grouped_callids()

        self._totlen = None
        self._chunkings = None

    @property
    def totlen(self):
        if self._totlen is None:
            self._read_chunkings()

        return self._totlen

    @property
    def chunkings(self):
        if self._chunkings is None:
            self._read_chunkings()

        return self._chunkings

    def _read_all_grouped_callids(self):
        grouped_callids = dict()

        # NOTE: Assuming labels and audios have the same group and dset structure
        # use audio root to visit all the groupids, and subsequent callids
        with h.File(self.filepath, 'r') as f:
            root = f[self.audios_root]

            for g in root.keys():  # groupids
                grouped_callids[g] = set(root[g].keys())  # callids

        return grouped_callids

    def _read_chunkings(self):
        # NOTE: We use the chunking info from the audios
        # and use the same for labels.
        chunkings = []
        total_len = 0

        with h.File(self.filepath, 'r') as f:
            a = f[self.audios_root]
            l = f[self.labels_root]

            for groupid in sorted(self.grouped_callids.keys()):
                for callid in sorted(self.grouped_callids[groupid]):
                    ad = a[groupid][callid]  # h5 Dataset
                    ld = l[groupid][callid]  # h5 Dataset

                    totlen = ad.shape[0]

                    starts = np.arange(0, totlen, ad.chunks[0])
                    ends = np.empty_like(starts)
                    ends[:-1] = starts[1:]
                    ends[-1] = totlen

                    total_len += totlen

                    chunkings.extend(
                        tu.Chunking(
                            datapath=ad.name,
                            dataslice=np.s_[s:e, ...],
                            labelpath=ld.name,
                            labelslice=np.s_[s:e, ...])
                        for s, e in zip(starts, ends))

        self._totlen = total_len
        self._chunkings = chunkings

    @classmethod
    def for_groupids(cls,
                     filepath,
                     groupids='all',
                     audios_root='audios',
                     labels_root='labels',
                     **kwargs):
        obj = cls(filepath,
                  audios_root=audios_root,
                  labels_root=labels_root,
                  **kwargs)

        if groupids == 'all':
            return obj

        if not isinstance(groupids, list):
            # only calls from the single groupid has to be kept
            obj.grouped_callids = {groupids: obj.grouped_callids[groupids]}
        else:
            # call callids from the specified groupids will be kept
            grouped_callids = dict()

            # NOTE: Doing like this to raise KeyError for incorrect groupids
            for g in groupids:
                grouped_callids[g] = obj.grouped_callids[g]

            obj.grouped_callids = grouped_callids

        return obj

    @classmethod
    def for_groupids_at(cls,
                        filepath,
                        at=np.s_[:],
                        audios_root='audios',
                        labels_root='labels',
                        **kwargs):
        obj = cls(filepath,
                  audios_root=audios_root,
                  labels_root=labels_root,
                  **kwargs)

        if at == np.s_[:]:
            return obj

        allgroupids = np.sort(list(obj.grouped_callids.keys()))

        try:
            groupids_at = allgroupids[at]
        except IndexError:
            print("\nTotal number of GroupIDs: {}\n".format(len(allgroupids)))
            raise

        if not isinstance(groupids_at, np.ndarray):
            # HACK: single group
            obj.grouped_callids = {
                groupids_at: obj.grouped_callids[groupids_at]
            }
        else:
            grouped_callids = dict()

            for g in groupids_at:
                grouped_callids[g] = obj.grouped_callids[g]

            obj.grouped_callids = grouped_callids

        return obj

    @classmethod
    def for_callids(cls,
                    filepath,
                    callids='all',
                    audios_root='audios',
                    labels_root='labels',
                    **kwargs):
        # FIXME: figure out proper way to kwargs
        obj = cls(filepath,
                  audios_root=audios_root,
                  labels_root=labels_root,
                  **kwargs)

        if callids == 'all':
            return obj

        allcallids = set.union(*obj.grouped_callids.values())
        if not isinstance(callids, list):
            if callids not in allcallids:
                raise ValueError("CallID {} not in file".format(callids))

            obj.grouped_callids = {groupid_for_callid(callids): {callids}}
        else:
            if not set(callids).issubset(allcallids):
                raise ValueError("Some CallIDs not in file")

            grouped_callids = dict()

            for c in callids:
                g = groupid_for_callid(c)

                v = grouped_callids.get(g, set())
                grouped_callids[g] = v.union({c})

            obj.grouped_callids = grouped_callids

        return obj

    @classmethod
    def for_callids_at(cls,
                       filepath,
                       at=np.s_[:],
                       audios_root='audios',
                       labels_root='labels',
                       **kwargs):
        obj = cls(filepath,
                  audios_root=audios_root,
                  labels_root=labels_root,
                  **kwargs)

        if at == np.s_[:]:
            return obj

        allcallids = np.sort(list(obj.grouped_callids.values()))

        try:
            callids_at = allcallids[at]
        except IndexError:
            print("\nTotal number of CallIDs: {}\n".format(len(allcallids)))
            raise

        if not isinstance(callids_at, np.ndarray):
            # HACK: single callid
            obj.grouped_callids = {
                groupid_for_callid(callids_at): {callids_at}
            }
        else:
            grouped_callids = dict()

            for c in callids_at:
                g = groupid_for_callid(c)

                v = grouped_callids.get(g, set())
                grouped_callids[g] = v.union({c})

            obj.grouped_callids = grouped_callids

        return obj


class FisherPerSamplePrepper(tu.BaseH5ChunkPrepper):
    """ Prep Fisher data, where each vector is an individual sample. No Context is added.
    - The data is normalized, if set to True, on a per-chunk basis
    - The label is normalized to nclasses to_categorical form, which can also be set
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            filepath,
            mean_it=True,
            std_it=True,
            nclasses=3,
            to_categorical=True,
            **kwargs):
        super(FisherPerSamplePrepper, self).__init__(filepath, **kwargs)
        self.mean_it = mean_it
        self.std_it = std_it
        self.nclasses = nclasses
        self.to_categorical = to_categorical

    def normalize_data(self, data):
        if self.mean_it:
            ndata = (data - data.mean(axis=0))
        else:
            ndata = data

        if self.std_it:
            ndata = ndata / data.std(axis=0)

        return ndata

    def prep_data(self, data):
        return self.normalize_data(data)

    def normalize_label(self, label):
        l = label.sum(axis=1).clip(min=0, max=self.nclasses - 1)
        if self.to_categorical:
            return nu.to_categorical(l, nclasses=self.nclasses, warn=False)
        else:
            return l

    def prep_label(self, label):
        return self.normalize_label(label)


class FisherPerSampleDataProvider(FisherH5ChunkingsReader,
                                  FisherPerSamplePrepper,
                                  tu.BaseInputsProvider):
    pass
