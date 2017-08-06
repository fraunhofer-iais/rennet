"""
@mosjust
Created: 05-08-2017

Keras Utilities
"""
from __future__ import print_function, division
import numpy as np
from os.path import join as pjoin
from keras.callbacks import Callback
from h5py import File as hFile

from rennet.utils.np_utils import (
    to_categorical, confusion_matrix_forcategorical,
    normalize_confusion_matrix, printoptions, print_prec_rec)


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
        gen = self.ip.flow(indefinitely=False, with_chunking=False)

        trues = []
        nsteps = 0
        for yy in gen:
            trues.append(yy[1])  # as per keras's expectations
            nsteps += 1

        trues = np.concatenate(trues)
        return trues, nsteps

    def _predict_calculate(self):
        gen = self.ip.flow(indefinitely=True, with_chunking=False)
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
                for path, data in zip(datas, paths):
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

        self._maybe_export(self.trues, "trues")
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
        self._maybe_export(res, paths)

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
        self._maybe_export(res, paths)

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

        self._maybe_export(self.trues, "trues")
        paths = [
            "{}/{}/{}".format(n, _pass - 1, _epoc - 1)
            for n in ['preds', 'confs', 'precs', 'recs']
        ]
        self._maybe_export(res, paths)
