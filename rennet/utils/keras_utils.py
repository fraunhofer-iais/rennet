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
            indefinitely=True, only_labels=False, with_chunking=False)
        preds = self.model.predict_generator(gen, self.nsteps, **self._kwargs)

        _preds = to_categorical(
            preds.argmax(axis=-1),
            nclasses=self.trues.shape[-1], )
        conf = confusion_matrix_forcategorical(self.trues, _preds)

        with np.errstate(invalid='ignore'):
            prec_rec = normalize_confusion_matrix(conf)

        return (preds, conf, ) + prec_rec

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
            print("percents  {}".format(100 * self.trues.sum(axis=0) /
                                        self.trues.sum()))

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
            _epoc, )
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
            checkpoints_pattern,
            save_best_only=False,
            save_weights_only=False,
            period=1,
            verbose=0, ),
        TensorBoard(
            log_dir=activity_dir,
            histogram_freq=
            1,  # FIXME: may not work with validation data as generator
            write_images=True,
            write_graph=False, ),
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
                    "{}/{}".format(p, currn)
                    for p in ('trues', 'preds', 'confs')
                ],
                datas=[t, p, conf], )

            currn = chunking.labelpath
            ctrue = ctrue[-1:]
            cpred = cpred[-1:]
