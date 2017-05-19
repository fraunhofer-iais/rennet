"""
@motjuste
Created: 06-02-2017

Utilities for training with Keras
"""
from __future__ import print_function, division
import numpy as np
import keras.callbacks as kc

import rennet.utils.np_utils as npu
import rennet.utils.plotting_utils as pu

to_categorical = npu.to_categorical

confusion_matrix = npu.confusion_matrix

normalize_confusion_matrix = npu.normalize_confusion_matrix

print_normalized_confusion = npu.print_normalized_confusion

plot_speclike = pu.plot_speclike

plot_normalized_confusion_matrix = pu.plot_normalized_confusion_matrix

plot_confusion_precision_recall = pu.plot_confusion_precision_recall


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

    def __init__(  # pylint: disable=too-many-arguments
            self,
            true_data,
            true_categorical_labels,
            print_on_end=True,
            plot_on_end=False,
            sr=8000,
            hop_sec=0.005, export_to=None):
        self.true_data = true_data
        self.true_label = true_categorical_labels
        self.plot_on_end = plot_on_end
        self.print_on_end = print_on_end
        self.sr = sr
        self.hop_sec = hop_sec
        self.export_to = export_to

        self.confusions = []
        self.confprec = None
        self.confrec = None
        self.last_preds = None  # last predictions from model

        self.nclasses = true_categorical_labels.shape[-1]
        self.colorcycle_for_true = ['grey', 'yellowgreen',
                                    'lightcoral']  #, 'lightskyblue']
        self.linecycle_for_TP_pred = ['-']
        self.linecycle_for_FP_pred = [':', '--', '-.']
        self.marker = '|'
        super(ConfusionHistory, self).__init__()

    def on_epoch_end(self, e, l=None):  # pylint: disable=unused-argument
        self.last_preds = self.model.predict(self.true_data, verbose=0, batch_size=1024)
        self._update_confusions()

    def on_train_end(self, l=None):  # pylint: disable=unused-argument
        self._set_conf_prec_rec()
        if self.print_on_end:
            print_normalized_confusion(
                self.confrec[-1, ...], title='RECALL CONFUSION')
            print_normalized_confusion(
                self.confprec[-1, ...], title='PRECISION CONFUSION')

        if self.plot_on_end:
            self.plot_last_normalized_confusions()
            self.plot_last_pred_classes()
            self.plot_confusions_history()

    def _set_conf_prec_rec(self):
        self.confprec, self.confrec = npu.normalize_confusion_matrices(
            np.array(self.confusions))

    @property
    def last_pred_classes(self):
        return np.argmax(self.last_preds, axis=-1)

    @property
    def last_pred_categorical(self):
        return to_categorical(self.last_pred_classes, nclasses=self.nclasses)

    def _update_confusions(self):
        self.confusions.append(
            npu.confusion_matrix_forcategorical(self.true_label, self.last_pred_categorical))

    def plot_last_normalized_confusions(  # pylint: disable=too-many-arguments
            self,
            perfigsize=(4, 4),
            cmap=pu.plt.cm.Blues,
            fontcolor='red',
            fontsize=16,
            figtitle='Last Confusion Matrix',
            subplot_titles=('Recall', 'Precision'),
            show=True,
            *args,
            **kwargs):
        plot_confusion_precision_recall(
            self.confrec[-1, ...],
            self.confprec[-1, ...],
            perfigsize=perfigsize,
            figtitle=figtitle,
            subplot_titles=subplot_titles,
            show=show,
            fontsize=fontsize,
            fontcolor=fontcolor,
            cmap=cmap,
            *args,
            **kwargs)

    def plot_last_pred_classes(  # pylint: disable=too-many-arguments
            self,
            figsize=(20, 4),
            show_time=True,
            sr=None,
            hop_sec=None,
            show=False):
        # TODO: [ ] Test. Check if there are any issues
        if sr is None:
            sr = self.sr

        if hop_sec is None:
            hop_sec = self.hop_sec

        plot_speclike(
            [self.last_pred_classes, np.argmax(self.true_label, axis=-1)],
            figsize=figsize,
            show_time=show_time,
            sr=sr,
            hop_sec=hop_sec,
            show=show)

    def linestyle_for_true_pred(self, true, pred):
        if true == pred:
            return self.linecycle_for_TP_pred[true %
                                              len(self.linecycle_for_TP_pred)]
        else:
            return self.linecycle_for_FP_pred[pred %
                                              len(self.linecycle_for_FP_pred)]

    def color_for_true_pred(self, true):
        return self.colorcycle_for_true[true % len(self.colorcycle_for_true)]

    def plot_confusions_history(self,
                                figsize=(20, 8),
                                fig_title="Confusions History"):
        if self.confusions is None:
            raise RuntimeError(
                "Confusion Matrices absent."
                "\nWas the callback added to the training function?")

        if self.confprec is None:  # if by chance on_train_end was never called
            # Expect confrec to be none as well
            self._set_conf_prec_rec()

        # NOTE: Doing the plotting here from scratch instead of pu.plot_multi
        # because doing custom setting of color and linestyle
        # There might be a way to pass a function to matplotlib, but...Lazy!
        # Plus, it is easy!
        fig, ax = pu.plt.subplots(2, 1, figsize=figsize)

        fig.suptitle(fig_title)

        _, trues, preds = self.confrec.shape
        titles = ('Recall', 'Precision')
        for i, con in enumerate([self.confrec, self.confprec]):
            for t in range(trues):
                for p in range(preds):
                    ax[i].plot(
                        con[:, t, p],
                        color=self.color_for_true_pred(t),
                        linestyle=self.linestyle_for_true_pred(t, p),
                        label="{}_{}".format(t, p),
                        marker=self.marker)
                    ax[i].set_ylim([0, 1])
            ax[i].legend()
            ax[i].set_title(titles[i])
            ax[i].grid()
