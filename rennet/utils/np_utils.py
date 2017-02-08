"""
@motjuste
Created: 28-09-2016

Numpy utilities
"""
import numpy as np


def group_by_values(values):
    # TODO: [A] add to a generic util module
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

    TODO: [ ] test for multi-sequence labels

    # Parameters
        y: 1D numpy class vector to be converted.
        nclasses: optional total number of classes

    # Returns
        A one-hot encodede class matrix

    # Raises
        RuntimeError: when any y is greater than nclasses
    """
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

    return (np.arange(nclasses) == y[:, None]).astype(np.float)


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
        _Ypred_i = Ypred[Ytrue[:, i].astype(
            np.bool), :]  # Preds for known class i
        for j in range(nclasses):
            conf[i, j] = np.sum(_Ypred_i[:, j])

    return conf


def confusion_matrix(ytrue, ypred, nclasses=None, warn=False):
    assert ytrue.shape == ypred.shape, "Shape mismatch: True {} != {} Predictions".format(
        ytrue.shape, ypred.shape)
    assert len(ytrue.shape) == 1, "Only supports vectors of class labels"

    Ytrue = to_categorical(ytrue, nclasses=nclasses, warn=warn)

    # Ytrue tells the correct nclasses now
    Ypred = to_categorical(ypred, nclasses=Ytrue.shape[-1], warn=warn)

    return categorical_confusion_matrix(Ytrue, Ypred)


def normalize_confusion_matrix(confusion_matrix):
    confprec, confrecall =  normalize_confusion_matrices(confusion_matrix[np.newaxis, ...])

    return confprec[0, ...], confrecall[0, ...]

def normalize_confusion_matrices(confusion_matrices):
    conf_sum1 = confusion_matrices.sum(axis=1)
    confprec = confusion_matrices / conf_sum1[np.newaxis, :]

    confrecall = confprec

    return confprec, confrecall
