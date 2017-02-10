"""
@motjuste
Created: 08-10-2016

Utilities for plotting
"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from librosa.display import specshow


def plot_multi(  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
        x_list,
        func="plot",
        rows=None,
        cols=4,
        perfigsize=(4, 4),
        subplot_titles=None,
        labels=None,
        fig_title=None,
        show=True,
        *args,
        **kwargs):
    if rows is None:
        rows = ceil(len(x_list) / cols)

    fgsz = (perfigsize[0] * cols, perfigsize[1] * rows)
    fig, ax = plt.subplots(rows, cols, figsize=fgsz)

    if len(x_list) == 1:
        ax = [ax]

    fig.suptitle(fig_title)

    at = lambda i: divmod(i, cols)
    if rows == 1:
        at = lambda i: i

    if labels is None:
        labels = [None for _ in range(len(x_list))]

    if subplot_titles is None:
        subplot_titles = list(range(len(x_list)))

    if func == "plot":
        for i, sx in enumerate(x_list):
            ax[at(i)].plot(sx, label=labels[i], *args, **kwargs)
    elif func == "pie":
        for i, sx in enumerate(x_list):
            ax[at(i)].pie(sx, labels=labels[i], *args, **kwargs)
            ax[at(i)].axis("equal")
    elif func == "hist":
        for i, sx in enumerate(x_list):
            ax[at(i)].hist(sx, *args, **kwargs)
    elif func == "imshow":
        for i, sx in enumerate(x_list):
            ax[at(i)].imshow(sx, *args, **kwargs)
    elif func == "confusion":
        # Find confusion specific kwargs, and pop them before forwarding

        # REF: http://stackoverflow.com/questions/11277432/how-to-remove-a-key-from-a-python-dictionary
        fontcolor = kwargs.pop('conf_fontcolor', None)
        fontcolor = 'red' if fontcolor is None else fontcolor

        fontsize = kwargs.pop('conf_fontsize', None)
        fontsize = 16 if fontsize is None else fontsize
        for i, sx in enumerate(x_list):
            # plotting the colors
            ax[at(i)].imshow(sx, interpolation='none', *args, **kwargs)
            ax[at(i)].set_aspect(1)

            # REF: http://stackoverflow.com/questions/20416609/remove-the-x-axis-ticks-while-keeping-the-grids-matplotlib
            ax[at(i)].set_xticklabels([])
            ax[at(i)].set_yticklabels([])

            ax[at(i)].set_xticks([0.5 + 1 * i for i in range(len(sx))])
            ax[at(i)].set_yticks([0.5 + 1 * i for i in range(len(sx))])
            ax[at(i)].grid(True, linestyle=':')

            # adding text for values
            w, h = sx.shape
            for x in range(w):
                for y in range(h):
                    ax[at(i)].annotate(
                        "{:.2f}%".format(sx[x][y] * 100),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color=fontcolor,
                        fontsize=fontsize)
    else:
        raise ValueError("Unsupported plotting function {}".format(func))

    # set title for subplot
    for i, st in enumerate(subplot_titles):
        ax[at(i)].set_title(st)

    if show:
        plt.show()


def plot_speclike(  # pylint: disable=too-many-arguments
        orderedlist,
        figsize=(20, 4),
        show_time=False,
        sr=16000,
        hop_sec=0.05,
        cmap=plt.cm.viridis,
        show=True):
    assert all(
        o.shape[0] == orderedlist[0].shape[0]
        for o in orderedlist), "All list items should be of the same length"

    x_axis = 'time' if show_time else None
    hop_len = int(hop_sec * sr)

    plt.figure(figsize=figsize)
    specshow(
        np.vstack(reversed(orderedlist)),
        x_axis=x_axis,
        sr=sr,
        hop_length=hop_len,
        cmap=cmap, )
    plt.colorbar()

    if show:
        plt.show()


def plot_normalized_confusion_matrix(  # pylint: disable=too-many-arguments
        confusion_matrix,
        figsize=(4, 4),
        cmap=plt.cm.Blues,
        fontcolor='red',
        fontsize=16,
        figtitle='Confusion Matrix',
        subplot_title="",
        show=True,
        *args,
        **kwargs):
    plot_multi(
        [confusion_matrix],
        func="confusion",
        rows=1,
        cols=1,
        perfigsize=figsize,
        fig_title=figtitle,
        subplot_titles=[subplot_title],
        show=show,
        # add these at end as part of kwargs
        conf_fontsize=fontsize,
        conf_fontcolor=fontcolor,
        cmap=cmap,
        *args,
        **kwargs)


def plot_confusion_precision_recall(  # pylint: disable=too-many-arguments
        conf_precision,
        conf_recall,
        perfigsize=(4, 4),
        cmap=plt.cm.Blues,
        fontcolor='red',
        fontsize=16,
        figtitle='Confusion Matrix',
        subplot_titles=('Precision', 'Recall'),
        show=True,
        *args,
        **kwargs):
    plot_multi(
        [conf_precision, conf_recall],
        func="confusion",
        rows=1,
        cols=2,
        perfigsize=perfigsize,
        fig_title=figtitle,
        subplot_titles=subplot_titles,
        show=show,
        # add these at end as part of kwargs
        conf_fontsize=fontsize,
        conf_fontcolor=fontcolor,
        cmap=cmap,
        *args,
        **kwargs)
