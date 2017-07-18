from __future__ import division, print_function
import os
import glob
import time
import h5py as h
import librosa as lr
import numpy as np
np.set_printoptions(formatter={'float': '{: 8.3f}'.format}, suppress=True)

import keras.layers as kl
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard

import rennet.datasets.fisher as fe
import rennet.utils.np_utils as nu
from rennet.utils.training_utils import BaseInputsProvider
from rennet.models.training_utils import ConfusionHistory


def read_fps(data_root, provider, dataset, export, featquerystr=None):
    pickles_dir = os.path.join(data_root, 'working', provider, dataset, export,
                               'pickles')

    if featquerystr is not None:
        raise NotImplementedError("custom querystr not yet supported")

    featdir = glob.glob(os.path.join(pickles_dir, "*"))[0]

    print("Searching in {} for pickles and h5".format(featdir))
    val_h5 = glob.glob(os.path.join(featdir, "*-val.h5"))[0]
    trn_h5 = glob.glob(os.path.join(featdir, "*-trn.h5"))[0]

    return val_h5, trn_h5


class CallLogAmper(fe.FisherPerSamplePrepper):
    def read_call_mean_std(self, chunking):
        with h.File(self.filepath, 'r') as f:
            mean = f[chunking.datapath].attrs['mean']
            std = f[chunking.datapath].attrs['std']

        return mean, std

    def get_prepped_data_label(self, chunking):
        data, label = self.read_h5_data_label_chunk(chunking)

        return self.prep_data(data, chunking), self.prep_label(label)

    def prep_data(self, data, chunking):
        mean, std = self.read_call_mean_std(chunking)
        if self.mean_it:
            ndata = lr.logamplitude(data, ref=mean)
        else:
            ndata = data

        if self.std_it:
            ndata = lr.logamplitude(data, ref=std)

        return ndata

class CallLogFBanker(fe.FisherPerSamplePrepper):
    def prep_data(self, data, chunking):
        mean, std = self.read_call_mean_std(chunking)
        nmels = 64
        data = lr.feature.melspectrogram(S=data.T, sr=8000, n_mels=nmels).T
        mean = lr.feature.melspectrogram(S=mean.T, sr=8000, n_mels=nmels).T
        std = lr.feature.melspectrogram(S=std.T, sr=8000, n_mels=nmels).T
        if self.mean_it:
            ndata = lr.logamplitude(data, ref=mean)
        else:
            ndata = data

        if self.std_it:
            ndata = lr.logamplitude(data, ref=std)

        return ndata
        

class SequenceLogAmper(CallLogAmper):
    def __init__(self, filepath, ctxframes=10, steps_per_chunk=1, **kwargs):
        super(SequenceLogAmper, self).__init__(filepath, **kwargs)
        self.ctx = ctxframes
        self.hctx = self.ctx // 2
        self.steps_per_chunk = steps_per_chunk

    def prep_data(self, data, chunking):
        ndata = super(SequenceLogAmper, self).prep_data(data, chunking)

        strided = nu.strided_view(ndata, 1 + 2 * self.ctx, 1)

        return strided[..., np.newaxis]

    def prep_label(self, label):
        label = super(SequenceLogAmper, self).prep_label(label)

        strided_l = nu.strided_view(label, 1 + 2 * self.ctx, 1)

        strided_l = strided_l[:, self.ctx - self.hctx:self.ctx + self.hctx + 1,
                              ...].argmax(axis=-1)

        return nu.to_categorical(
            strided_l.max(axis=-1), nclasses=3, warn=False)

    @classmethod
    def maybe_shuffle(self, arr, shuffle_seed):
        #         indices = super(SequenceLogAmper, self).maybe_shuffle(np.arange(len(arr)), shuffle_seed)
        # TODO: How to shuffle without copying the arrays?
        # no, fancy indexing with shuffled indices does make copies
        #         if shuffle_seed is not None:
        #             fe.warnings.warn("SequenceLogAmper doesn't support shuffling for now")
        return arr

    def get_prepped_inputs(self, chunking, shuffle_seed=None, **kwargs):  # pylint: disable=unused-argument
        d, l = super(SequenceLogAmper, self).get_prepped_inputs(
            chunking, shuffle_seed=shuffle_seed, **kwargs)

        stepsize = 1 + len(d) // self.steps_per_chunk
        start = 0

        while start < len(d):
            ss = np.s_[start:start + stepsize, ...]

            yield d[ss], l[ss]
            start += stepsize

            
class SequenceLogFBanker(SequenceLogAmper, CallLogFBanker):
    pass

def only_splw_2(labels):
    lsums = labels.sum(axis=0)
    if lsums[-1] == 0:
        return None

    else:
        _clsw = lsums.max() / lsums
        splw = np.ones(shape=len(labels))
        splw[labels[:, 2] == 1] = _clsw[2]

        return splw


def clsw(labels, clsw_tuple=(1, 1, 1)):
    splw = np.ones(shape=len(labels))
    for i, w in enumerate(clsw_tuple):
        splw[labels[:, i] == 1] = w

    return splw


def ones(labels):
    return np.ones(len(labels))


class FisherSeqSkippingLogAmperDP(fe.FisherH5ChunkingsReader, SequenceLogAmper,
                                  BaseInputsProvider):
    def __init__(self, filepath, splwfn=ones, **kwargs):
        super(FisherSeqSkippingLogAmperDP, self).__init__(filepath, **kwargs)

        self.splwfn = splwfn

    def flow_for_epoch_at(self, at):
        gen = super(FisherSeqSkippingLogAmperDP, self).flow_for_epoch_at(at)

        for x in gen:
            for data, labels in x:

                splw = self.splwfn(labels)
                if splw is None:
                    continue
                else:
                    yield data, labels, splw


class FisherSeqSkippingLogFBankerDP(fe.FisherH5ChunkingsReader, SequenceLogFBanker,
                                  BaseInputsProvider):
    def __init__(self, filepath, splwfn=ones, **kwargs):
        super(FisherSeqSkippingLogFBankerDP, self).__init__(filepath, **kwargs)

        self.splwfn = splwfn

    def flow_for_epoch_at(self, at):
        gen = super(FisherSeqSkippingLogFBankerDP, self).flow_for_epoch_at(at)

        for x in gen:
            for data, labels in x:

                splw = self.splwfn(labels)
                if splw is None:
                    continue
                else:
                    yield data, labels, splw

                    

class ChattyConfHist(ConfusionHistory):
    def __init__(self, *args, **kwargs):
            super(ChattyConfHist, self).__init__(*args, **kwargs)
            self._curr_e = 0

    def on_batch_end(self, b, l=None):
        if b % 800 == 0:
            super(ChattyConfHist, self).on_epoch_end(b, l)
            self._set_conf_prec_rec()
            print("b-{:6} P(REC)".format(b), "  ".join(
                "{:6.2f} ({:6.2f})".format(p, r)
                for r, p in zip(self.confrec[-1, ...].diagonal() * 100,
                                self.confprec[-1, ...].diagonal() * 100)))

            if self.export_to is not None:
                with h.File(self.export_to, 'a') as f:
                    if 'trues' not in f.keys():
                        f.create_dataset('trues', data=self.true_label)

                    f.create_dataset('preds/e/b/{}_{}'.format(self._curr_e, b), data=self.last_preds)
                    f.create_dataset('confs/e/b/{}_{}'.format(self._curr_e, b), data=self.confusions[-1])

            time.sleep(0)

    def on_epoch_end(self, e, l=None):
        super(ChattyConfHist, self).on_epoch_end(e, l)
        self._set_conf_prec_rec()

        print()
        print("e-{:6} P(REC)".format(e), "  ".join(
            "{:6.2f} ({:6.2f})".format(p, r)
            for r, p in zip(self.confrec[-1, ...].diagonal() * 100,
                            self.confprec[-1, ...].diagonal() * 100)))
        print()

        if self.export_to is not None:
            with h.File(self.export_to, 'a') as f:
                if 'trues' not in f.keys():
                    f.create_dataset('trues', data=self.true_label)

                f.create_dataset('preds/e/{}'.format(e), data=self.last_preds)
                f.create_dataset('confs/e/{}'.format(e), data=self.confusions[-1])
		
        self._curr_e += 1

        time.sleep(0)


def create_callbacks(activity_dir, checkpoints_fn, Xval, Yval):
    callbacks = []

    callbacks.append(
        ModelCheckpoint(checkpoints_fn, monitor='val_loss', verbose=0))
    callbacks.append(
        TensorBoard(
            log_dir=activity_dir,
            histogram_freq=1,
            write_images=True,
            write_graph=False, ))
    callbacks.append(
        ChattyConfHist(
            Xval, Yval, export_to=os.path.join(activity_dir,
                                               'confhistory.h5')))

    return callbacks


def train_eval(model, trn_dp, steps_per_epoch, epochs, *args, **kwargs):
    model.fit_generator(
        trn_dp.flow(indefinitely=True),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        *args,
        **kwargs)
