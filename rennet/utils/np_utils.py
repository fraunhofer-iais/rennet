"""
@motjuste
Created: 28-09-2016

Numpy utilities
"""
from __future__ import division
import numpy as np


def group_by_values(values):
    # TODO: [A] make it work with 1 dimensional arrays
    # Ref: http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance

    initones = [[1] * values.shape[1]]
    diff = np.concatenate([initones, np.diff(values, axis=0)])
    starts = np.unique(np.where(diff)[0])  # remove duplicate starts

    ends = np.ones(len(starts), dtype=np.int)
    ends[:-1] = starts[1:]
    ends[-1] = len(values)

    labels = values[starts]

    return np.vstack([starts, ends]).T, labels


def to_categorical(y, nclasses=None, warn=False):
    """ Convert class vectors to one-hot class matrix

    TODO: [ ] check if works for n-dim sequences
    # Parameters
        y: numpy class vector to be converted.
            expected shape: (Epoch, Batch, Sequence), BUT
            as long as the last dimension has the class label, it should work
            please report weird behavior.
            NOTE: multi-dim Sequence not tested
        nclasses: optional total number of classes

    # Returns
        A one-hot encodede class matrix

    # Raises
        RuntimeError: when any y is greater than nclasses
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    ymax = np.max(y) + 1  # zero is a class
    if nclasses is None:
        nclasses = ymax
    elif nclasses < ymax:
        raise RuntimeError(
            "Some class labels are greater than provided nclasses: {} > {}".
            format(ymax, nclasses))
    elif nclasses > ymax and warn:
        raise RuntimeWarning(
            "Some class labels may be missing: {} > {}".format(nclasses, ymax))

    res = np.arange(nclasses)[np.newaxis, :] == y[..., np.newaxis]
    res = res.astype(np.float)  # cuz it is a probability dist

    return res


def strided(x, nperseg, noverlap):
    """ Create strided view of array without copying

    NOTE: Striding happens in the last dimension of multidimensional input

    TODO: [ ] Add proper tests for multidimensional striding
    TODO: [ ] Check allowing choice of striding dimension
    """
    step = nperseg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])

    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def categorical_confusion_matrix(Ytrue, Ypred):
    assert Ytrue.shape == Ypred.shape, "Shape mismatch: True {} != {} Predictions".format(
        Ytrue.shape, Ypred.shape)
    assert len(Ytrue.shape) == 2, "Only supports vectors of categorical labels"

    nclasses = Ytrue.shape[-1]

    conf = np.zeros(shape=(nclasses, nclasses), dtype=np.int)

    for i in range(nclasses):
        # Preds for known class i
        _Ypred_i = Ypred[Ytrue[:, i].astype(np.bool), :]
        for j in range(nclasses):
            conf[i, j] = np.sum(_Ypred_i[:, j])

    return conf


def confusion_matrix(ytrue, ypred, nclasses=None, warn=False):
    if not isinstance(ytrue, np.ndarray):
        ytrue = np.array(ytrue)

    if not isinstance(ypred, np.ndarray):
        ypred = np.array(ypred)

    assert ytrue.shape == ypred.shape, "Shape mismatch: True {} != {} Predictions".format(
        ytrue.shape, ypred.shape)
    assert len(ytrue.shape) == 1, (
        "Only supports vectors of class labels. "
        "If your labels are one-hot, please use categorical_confusion_matrix")

    Ytrue = to_categorical(ytrue, nclasses=nclasses, warn=warn)

    # Ytrue tells the correct nclasses now
    Ypred = to_categorical(ypred, nclasses=Ytrue.shape[-1], warn=warn)

    return categorical_confusion_matrix(Ytrue, Ypred)


def categorical_confusion_matrices(Ytrue, Ypreds):
    assert Ytrue.shape == Ypreds.shape[
        1:], "Shape mismatch: True {} != {} Predictions".format(Ytrue.shape,
                                                                Ypreds.shape)
    assert len(Ytrue.shape) == 2, "Only supports vectors of categorical labels"

    nclasses = Ytrue.shape[-1]

    conf = np.zeros(shape=(Ypreds.shape[0], nclasses, nclasses), dtype=np.int)

    for i in range(nclasses):
        # Preds for known class i
        _Ypred_i = Ypreds[:, Ytrue[..., i].astype(np.bool), :]
        for j in range(nclasses):
            conf[:, i, j] = np.sum(_Ypred_i[..., j], axis=-1)

    return conf


def confusion_matrices(ytrue, ypreds, nclasses=None, warn=False):
    """ Calculating confusion matrices for multiple predictions.

    One true label, in vector of class number format.
    Multiple predictions of the same format.
    """
    if not isinstance(ytrue, np.ndarray):
        ytrue = np.array(ytrue)

    if not isinstance(ypreds, np.ndarray):
        ypreds = np.array(ypreds)

    assert ytrue.shape == ypreds.shape[
        1:], "Shape mismatch: True {} != {} Predictions".format(ytrue.shape,
                                                                ypreds.shape)

    assert len(ytrue.shape) == 1, "Only supports vectors of class labels"

    Ytrue = to_categorical(ytrue, nclasses=nclasses, warn=warn)

    # Ytrue tells the correct nclasses now
    Ypreds = to_categorical(ypreds, nclasses=Ytrue.shape[-1], warn=warn)

    return categorical_confusion_matrices(Ytrue, Ypreds)


def normalize_confusion_matrix(conf_matrix):
    if not isinstance(conf_matrix, np.ndarray):
        conf_matrix = np.array(conf_matrix)

    confmat = conf_matrix[np.newaxis, ...]
    prec, recall = normalize_confusion_matrices(confmat)

    return prec[0, ...], recall[0, ...]


def normalize_confusion_matrices(conf_matrices):
    if not isinstance(conf_matrices, np.ndarray):
        conf_matrices = np.array(conf_matrices)

    conf_sum = conf_matrices.sum(axis=-2)
    confprec = conf_matrices / conf_sum[:, np.newaxis]

    conf_sum = conf_matrices.sum(axis=-1)
    confrec = np.transpose(conf_matrices, [0, 2, 1]) \
              / conf_sum[:, np.newaxis]
    confrec = np.transpose(confrec, [0, 2, 1])

    return confprec, confrec


def print_normalized_confusion(confmat, title='CONFUSION MATRIX'):
    print("\n{:/>70}//".format(" {} ".format(title)))
    print(np.round(confmat * 100, decimals=2))
