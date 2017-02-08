"""
@motjuste
Created: 06-02-2017

Utilities for training with Keras
"""
from __future__ import print_function, division
import numpy as np
import keras.callbacks as kc

import rennet.utils.np_utils as npu

to_categorical = npu.to_categorical

confusion_matrix = npu.confusion_matrix

normalize_confusion_matrix = npu.normalize_confusion_matrix


class ConfusionHistory(kc.Callback):  # pylint: disable=too-many-instance-attributes
    """ Callback class to store the confusion matrices per epoch

    Although the callbacks are essentially associated with a training session,
    this callback is more an encapsulation for doing stuff with the model's
    predictions, mainly the confusion matrix.

    The confusion matrix is calculated per epoch on the data provided at the
    time of initialization.

    It provides helper functions to calculate some stats without initialization
    It also provides helpers for plotting some of them.
    """

    def __init__(self,
                 true_data,
                 true_categorical_labels,
                 plot_on_end=False,
                 print_on_end=True):
        self.true_data = true_data
        self.true_label = true_categorical_labels
        self.nclasses = true_categorical_labels.shape[-1]
        self.plot_on_end = plot_on_end
        self.print_on_end = print_on_end
        self.confusions = []
        self.confprec = None
        self.confrec = None
        self.last_preds = None  # last predictions from model

        super(ConfusionHistory, self).__init__()

    def on_epoch_end(self, e, l=None):  # pylint: disable=unused-argument
        self.last_preds = self.model.predict(self.true_data, verbose=0)
        self._update_confusions()

    def on_train_end(self, l=None):  # pylint: disable=unused-argument
        self._set_conf_prec_rec()
        raise NotImplementedError()

    def _set_conf_prec_rec(self):
        self.confprec, self.confrec = npu.normalize_confusion_matrices(
            np.array(self.confusions))

    @property
    def last_pred_classes(self):
        return np.argmax(self.last_preds)

    @property
    def last_pred_categorical(self):
        return to_categorical(self.last_pred_classes, nclasses=self.nclasses)

    def _update_confusions(self):
        self.confusions.append(
            npu.categorical_confusion_matrix(self.true_label,
                                             self.last_pred_categorical))

    def plot_confusions_history(self):
        if self.confusions is None:
            raise RuntimeError(
                "Confusion Matrices absent."
                "\nWas the callback added to the training function?")

        if self.confprec is None:  # if by chance on_train_end was never called
            # Expect confrec to be none as well
            self._set_conf_prec_rec()

        raise NotImplementedError()
