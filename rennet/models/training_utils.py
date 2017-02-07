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


class ConfusionHistory(kc.Callback):
    """ Callback class to store the confusion matrices per epoch

    Although the callbacks are essentially associated with a training session,
    this callback is more an encapsulation for doing stuff with the model's
    predictions, mainly the confusion matrix.

    The confusion matrix is calculated per epoch on the data provided at the
    time of initialization.

    It provides helper functions to calculate some stats without initialization
    It also provides helpers for plotting some of them.
    """

    def __init__(self, true_data, true_categorical_labels):
        self.true_data = true_data
        self.true_label = true_categorical_labels
        self.nclasses = true_categorical_labels.shape[-1]
        self.confusions = []
        self.last_preds = None  # last predictions from model

        super(ConfusionHistory, self).__init__()

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
