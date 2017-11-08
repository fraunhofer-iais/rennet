"""
@mosjuste
Created: 05-08-2017

Keras Utilities
"""
from __future__ import print_function, division
import numpy as np
from os.path import join as pjoin
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from h5py import File as hFile

from keras.models import Sequential, Model
import keras.layers as kl

from rennet.utils.np_utils import (
    to_categorical, confusion_matrix_forcategorical,
    normalize_confusion_matrix, printoptions, print_prec_rec)


# CALLBACKS ####################################################### CALLBACKS #
class ChattyConfusionHistory(Callback):
    """ Callback to store and pretty-print confusion matrix per epoch.

    It accepts only a valid BaseInputsProvider to read data from.
    Check the relevant module for more details.

    NOTE: Expects Softmax/categorical labels from the model/inputs_provider
    """

    def __init__(self,
                 inputs_provider,
                 epochs_per_pass=1,
                 export_dir=None,
                 **kwargs):
        # IDEA: Accept numpy arrays as X and Y
        self.ip = inputs_provider
        self._kwargs = kwargs
        self.epp = int(epochs_per_pass)
        assert self.epp >= 1, "epochs_per_pass should be >= 1, v/s {}".format(
            epochs_per_pass)

        if export_dir is not None:
            self.export_to = pjoin(export_dir, "confusions.h5")
        else:
            self.export_to = None

        self.trues, self.nsteps = self._read_trues()
        self.prefixtr = "{:<9} "

        super(ChattyConfusionHistory, self).__init__()

    def _read_trues(self):
        gen = self.ip.flow(
            indefinitely=False, only_labels=True, with_chunking=False)

        trues = []
        nsteps = 0
        for yy in gen:
            trues.append(yy[1])  # as per keras's expectations
            nsteps += 1

        trues = np.concatenate(trues)
        return trues, nsteps

    def _predict_calculate(self):
        gen = self.ip.flow(
            indefinitely=True,
            only_labels=False,
            only_data=True,
            with_chunking=False)
        preds = self.model.predict_generator(gen, self.nsteps, **self._kwargs)

        _preds = to_categorical(
            preds.argmax(axis=-1),
            nclasses=self.trues.shape[-1],
        )
        conf = confusion_matrix_forcategorical(self.trues, _preds)

        with np.errstate(invalid='ignore'):
            prec_rec = normalize_confusion_matrix(conf)

        return (
            preds,
            conf,
        ) + prec_rec

    def _maybe_export(self, datas, paths, multi=False):
        if self.export_to is not None:
            if not multi:
                datas = [datas]
                paths = [paths]

            with hFile(self.export_to, 'a') as f:
                for path, data in zip(paths, datas):
                    if path not in f.keys():
                        f.create_dataset(
                            path,
                            data=data,
                            compression='lzf',
                            fletcher32=True)

                f.flush()

    def _print_class_stats(self):
        with printoptions(
                suppress=True,
                formatter={
                    'float': '{: >9.2f}'.format,
                    'int': '{: >9d}'.format
                }):
            print("Confusions on {}".format(len(self.trues)))
            print("per class {}".format(self.trues.sum(axis=0).astype(int)))
            print("percents  {}".format(
                100 * self.trues.sum(axis=0) / self.trues.sum()))

    def on_train_begin(self, *args, **kwargs):  # pylint: disable=unused-argument
        res = self._predict_calculate()

        self._maybe_export(self.trues, "trues", multi=False)
        print()
        self._print_class_stats()
        print()

        print(self.prefixtr.format('INITIAL'))
        print_prec_rec(*res[-2:], onlydiag=False)
        print()

        paths = [
            "initial/{}".format(n)
            for n in ['preds', 'confs', 'precs', 'recs']
        ]
        self._maybe_export(res, paths, multi=True)

        print(self.prefixtr.format('INITIAL'))
        print_prec_rec(*res[-2:], onlydiag=True)

    def on_train_end(self, *args, **kwargs):  # pylint: disable=unused-argument
        res = self._predict_calculate()

        print()
        self._print_class_stats()
        print()

        print(self.prefixtr.format('FINAL'))
        print_prec_rec(*res[-2:], onlydiag=False)
        print()

        paths = [
            "final/{}".format(n) for n in ['preds', 'confs', 'precs', 'recs']
        ]
        self._maybe_export(res, paths, multi=True)

    def on_epoch_end(self, e, *args, **kwargs):  # pylint: disable=unused-argument
        res = self._predict_calculate()

        _pass = 1 + e // self.epp
        _epoc = 1 + e % self.epp

        pre = "{}-{:>3}-{:>3}".format(
            'e' if _epoc < self.epp else 'p',
            _pass,
            _epoc,
        )
        print(self.prefixtr.format(pre), end='')
        print_prec_rec(*res[-2:], onlydiag=True)
        print()

        paths = [
            "{}/{}/{}".format(n, _pass - 1, _epoc - 1)
            for n in ['preds', 'confs', 'precs', 'recs']
        ]
        self._maybe_export(res, paths, multi=True)


model_checkpoint_pattern = 'w.{epoch:03d}-{val_loss:.3f}-{val_categorical_accuracy:.3f}.h5'


def create_callbacks(inputs_provider,
                     activity_dir,
                     epochs_per_pass,
                     checkpoints_pattern=model_checkpoint_pattern,
                     **kwargs):
    return [
        ModelCheckpoint(
            pjoin(activity_dir, checkpoints_pattern),
            save_best_only=False,
            save_weights_only=False,
            period=1,
            verbose=0,
        ),
        TensorBoard(
            log_dir=activity_dir,
            histogram_freq=1,  # FIXME: may not work with val data as generator
            write_images=True,
            write_graph=False,
        ),
        ChattyConfusionHistory(
            inputs_provider,
            epochs_per_pass=epochs_per_pass,
            export_dir=activity_dir,
            **kwargs),
        # IDEA: A predictions saver at the end of training?
    ]


def predict_on_inputs_provider(model, inputs_provider, export_to_dir,
                               **kwargs):
    export_to = pjoin(export_to_dir, "predictions.h5")

    def _save(paths, datas):
        with hFile(export_to, 'a') as f:
            for path, data in zip(paths, datas):
                if path not in f.keys():
                    f.create_dataset(
                        path, data=data, compression='lzf', fletcher32=True)

            f.flush()

    currn = None
    ctrue = []
    cpred = []
    for xy, (_, chunking) in inputs_provider.flow(
            indefinitely=False, only_labels=False, with_chunking=True,
            **kwargs):

        ctrue.append(xy[1])
        cpred.append(model.predict_on_batch(xy[0]))

        if currn is None:
            currn = chunking.labelpath
            continue

        if chunking.labelpath != currn:
            t = np.concatenate(ctrue[:-1])
            p = np.concatenate(cpred[:-1])
            conf = confusion_matrix_forcategorical(t,
                                                   to_categorical(
                                                       p.argmax(axis=-1),
                                                       nclasses=p.shape[-1]))

            _save(
                paths=[
                    "{}/{}".format(_p, currn)
                    for _p in ('trues', 'preds', 'confs')
                ],
                datas=[t, p, conf],
            )

            print(currn, end=' ')
            print_prec_rec(*normalize_confusion_matrix(conf), onlydiag=True)
            currn = chunking.labelpath
            ctrue = ctrue[-1:]
            cpred = cpred[-1:]


# MODELS ############################################################# MODELS #


def model_c3(input_shape, nclasses, compile_model=True):
    model = Sequential(name='conv3')

    # first conv2d layer
    model.add(
        kl.Conv2D(
            64,
            3,
            strides=1,
            data_format='channels_last',
            input_shape=input_shape[1:],
            name='c1_3_64_1',
        ))
    model.add(kl.BatchNormalization(name='c1_bn'))
    model.add(kl.Activation('relu', name='c1_relu'))
    model.add(kl.Dropout(0.1, name='c1_d_10'))
    model.add(kl.MaxPool2D(2, name='c1_mxp2_2'))

    # second conv2d layer
    model.add(
        kl.Conv2D(
            128,
            3,
            strides=1,
            data_format='channels_last',
            input_shape=input_shape[1:],
            name='c2_3_128_1',
        ))
    model.add(kl.BatchNormalization(name='c2_bn'))
    model.add(kl.Activation('relu', name='c2_relu'))
    model.add(kl.Dropout(0.1, name='c2_d_10'))
    model.add(kl.MaxPool2D(2, name='c2_mxp2_2'))

    # third conv2d layer
    model.add(
        kl.Conv2D(
            256,
            3,
            strides=1,
            data_format='channels_last',
            input_shape=input_shape[1:],
            name='c3_3_256_1',
        ))
    model.add(kl.BatchNormalization(name='c3_bn'))
    model.add(kl.Activation('relu', name='c3_relu'))
    model.add(kl.Dropout(0.1, name='c3_d_10'))

    # max globally
    model.add(kl.GlobalMaxPool2D(name='gmxp'))

    # first FC
    model.add(kl.Dense(512, activation='relu', name='f1_512_relu'))
    model.add(kl.Dropout(0.1, name='f1_d_10'))

    # second FC
    model.add(kl.Dense(128, activation='relu', name='f2_128_relu'))
    model.add(kl.Dropout(0.1, name='f2_d_10'))

    # second FC
    model.add(kl.Dense(32, activation='relu', name='f3_32_relu'))
    model.add(kl.Dropout(0.1, name='f3_d_10'))

    # output layer
    model.add(kl.Dense(nclasses, activation='softmax', name='sfmx'))

    if compile_model:
        # Compile and send the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adamax',
            metrics=['categorical_accuracy'],
        )

    return model


def model_c3_avg(input_shape, nclasses, compile_model=True):
    model = Sequential(name='conv3')

    # first conv2d layer
    model.add(
        kl.Conv2D(
            64,
            3,
            strides=1,
            data_format='channels_last',
            input_shape=input_shape[1:],
            name='c1_3_64_1',
        ))
    model.add(kl.BatchNormalization(name='c1_bn'))
    model.add(kl.Activation('relu', name='c1_relu'))
    model.add(kl.Dropout(0.1, name='c1_d_10'))
    model.add(kl.AvgPool2D(2, name='c1_avp2_2'))

    # second conv2d layer
    model.add(
        kl.Conv2D(
            128,
            3,
            strides=1,
            data_format='channels_last',
            input_shape=input_shape[1:],
            name='c2_3_128_1',
        ))
    model.add(kl.BatchNormalization(name='c2_bn'))
    model.add(kl.Activation('relu', name='c2_relu'))
    model.add(kl.Dropout(0.1, name='c2_d_10'))
    model.add(kl.AvgPool2D(2, name='c2_avp2_2'))

    # third conv2d layer
    model.add(
        kl.Conv2D(
            256,
            3,
            strides=1,
            data_format='channels_last',
            input_shape=input_shape[1:],
            name='c3_3_256_1',
        ))
    model.add(kl.BatchNormalization(name='c3_bn'))
    model.add(kl.Activation('relu', name='c3_relu'))
    model.add(kl.Dropout(0.1, name='c3_d_10'))

    # max globally
    model.add(kl.GlobalAvgPool2D(name='gavp'))

    # first FC
    model.add(kl.Dense(512, activation='relu', name='f1_512_relu'))
    model.add(kl.Dropout(0.1, name='f1_d_10'))

    # second FC
    model.add(kl.Dense(128, activation='relu', name='f2_128_relu'))
    model.add(kl.Dropout(0.1, name='f2_d_10'))

    # second FC
    model.add(kl.Dense(32, activation='relu', name='f3_32_relu'))
    model.add(kl.Dropout(0.1, name='f3_d_10'))

    # output layer
    model.add(kl.Dense(nclasses, activation='softmax', name='sfmx'))

    if compile_model:
        # Compile and send the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adamax',
            metrics=['categorical_accuracy'],
        )

    return model


def combine_keras_models_parallel(models, optimizer=None):
    """ Combine multiple **compiled** keras models into 1, all being in parallel.

    NOTE: Keras only supports one optimizer per model. If none is provided as argument,
    the optimizer of the first one will be used in the final model
    """
    # FIXME: what if any of the models is not built?
    assert all(model.built for model in
               models), "All models must have been compiled before merging"
    inputs = []
    outputs = []
    loss = []
    metrics = dict()
    for model in models:
        inputs.append(kl.Input(model.input_shape[1:]))
        outputs.append(model(inputs[-1]))

        loss.append(model.loss)
        metrics[outputs[-1]] = model.metrics

        if optimizer is None:
            optimizer = model.optimizer

    model = Model(inputs, outputs)
    model.compile(optimizer, loss, metrics=metrics)

    return model
