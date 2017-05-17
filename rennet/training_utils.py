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


def make_output_dirs(out_root, activity_name):
    activity_dir = os.path.join(out_root, activity_name)
    checkpoints_dir = os.path.join(activity_dir, 'checkpoints')
    checkpoints_fn = os.path.join(
        checkpoints_dir,
        'w.{epoch:04d}-{val_loss:.3f}-{val_categorical_accuracy:.3f}.h5')

    os.makedirs(activity_dir)
    os.makedirs(checkpoints_dir)

    return activity_dir, checkpoints_dir, checkpoints_fn


def read_fps(data_root, provider, dataset, export, featquerystr=None):
    pickles_dir = os.path.join(data_root, 'data', 'working', provider, dataset,
                               export, 'pickles')

    if featquerystr is not None:
        raise NotImplementedError("custom querystr not yet supported")
    else:
        featdir = glob.glob(os.path.join(pickles_dir, "*"))[0]

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


def only_splw_2(labels):
    lsums = labels.sum(axis=0)
    if np.any(lsums == 0):
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
    def __init__(self,
                 filepath,
                 shuffle_seed=None,
                 nepochs=1,
                 splwfn=ones,
                 **kwargs):
        super(FisherSeqSkippingLogAmperDP, self).__init__(filepath, shuffle_seed, nepochs, **kwargs)

        self.splwfn = splwfn

    def flow_for_epoch_at(self, at):
        gen = super(FisherSeqSkippingLogAmperDP, self).flow_for_epoch_at(at)

        for x in gen:
            for data, labels in x:

                splw = self.splwfn(labels)
                if splw is None:
                    continue
                else:
                    # lsums = labels.sum(axis=0)
                    # clsw = lsums.max() / lsums
                    # splw = np.ones(shape=(len(labels), ), dtype=clsw.dtype)
                    # # splw[labels[:, 0] == 1] = clsw[0]
                    # splw[labels[:, 2] == 1] = clsw[2]

                    yield data, labels, splw


def get_model(input_shape, nclasses=3):
    model = Sequential()
    model.add(
        kl.Conv2D(
            32,
            3,
            strides=1,
            activation='relu',
            name='block1_conv',
            data_format='channels_last',
            input_shape=input_shape[1:]))
    model.add(kl.BatchNormalization(
        name='block1_bn', ))
    model.add(kl.Dropout(
        0.2,
        name='block1_dp', ))

    model.add(kl.MaxPool2D(2))

    model.add(
        kl.Conv2D(64, 3, strides=1, activation='relu', name='block2_conv'))
    model.add(kl.BatchNormalization(name='block2_bn'))
    model.add(kl.Dropout(
        0.2,
        name='block2_dp', ))

    model.add(kl.MaxPool2D(2))

    model.add(
        kl.Conv2D(64, 3, strides=1, activation='relu', name='block3_conv'))
    model.add(kl.BatchNormalization(name='block3_bn'))
    model.add(kl.Dropout(
        0.2,
        name='block3_dp', ))

    model.add(kl.GlobalMaxPool2D(name='g_maxpool'))

    w = 256
    for i in range(2):
        model.add(kl.Dropout(0.2, name='block4_dp_{}'.format(i)))
        w = w // (4)
        model.add(
            kl.Dense(w, activation='relu', name='block4_fc{}_{}'.format(w, i)))

    model.add(kl.Dense(nclasses, activation='softmax', name='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])

    return model


class ChattyConfHist(ConfusionHistory):
    def on_epoch_end(self, e, l=None):
        super(ChattyConfHist, self).on_epoch_end(e, l)
        self._set_conf_prec_rec()
        print()
        print("P(REC)", "  ".join(
            "{:6.2f} ({:6.2f})".format(p, r)
            for r, p in zip(self.confrec[-1, ...].diagonal() * 100,
                            self.confprec[-1, ...].diagonal() * 100)))
        print()

        if self.export_to is not None:
            with h.File(self.export_to, 'a') as f:
                if f['trues'] not in f.keys():
                    f['trues'] = self.true_label

                f['preds/{}'.format(e)] = self.last_preds
                f['confs/{}'.format(e)] = self.confusions

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
            write_graph=False))
    callbacks.append(
        ChattyConfHist(
            Xval, Yval, export_to=os.path.join(activity_dir, 'confhistory')))

    return callbacks


def train_eval(model, trn_dp, steps_per_epoch, epochs, *args, **kwargs):
    model.fit_generator(
        trn_dp.flow(indefinitely=True),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        *args,
        **kwargs)
