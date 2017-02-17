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


def _generic_confusion_matrix_forcat(Ytrue,
                                     predicate_true,
                                     Ypred,
                                     predicate_pred,
                                     predicate_match,
                                     reduce_function,
                                     reduce_axis=None,
                                     reduce_keepdims=False):
    if not isinstance(Ytrue, np.ndarray):
        Ytrue = np.array(Ytrue)

    if not isinstance(Ypred, np.ndarray):
        Ypred = np.array(Ypred)

    # TODO: [ ] Add proper assertions to avoid obvious mistakes

    # assert Ytrue.shape == Ypred.shape, "Shape mismatch: "\
    #     "True {} != {} Predictions".format(Ytrue.shape, Ypred.shape)
    # NOTE: Can't use this, cuz case of multi-pred

    _valid_axes = [
        i for i in range(len(Ypred.shape) - 1) if Ypred.shape[i] > 1
    ]
    if reduce_axis is None:
        # choose the last axis > 1, excluding the ClassLabel axis
        # will raise error if none qualify, since then max is looking into empty
        reduce_axis = max(_valid_axes)
    else:
        if reduce_axis not in _valid_axes:
            msg = """The axis argument cannot be:
            - The last axis (axis of class label) {}
            - An axis of size <= 1 {}""".format(
                "TRUE" if reduce_axis == len(Ypred.shape) - 1 else "", "TRUE"
                if Ypred.shape[reduce_axis] <= 1 else "")
            raise ValueError(msg)

    conf = reduce_function(
        predicate_match(
            predicate_true(Ytrue[..., np.newaxis]),
            predicate_pred(Ypred[..., np.newaxis, :])),
        axis=reduce_axis,
        keepdims=reduce_keepdims)

    return conf


def confusion_matrix_forcategorical(Ytrue, Ypred, axis=None, keepdims=False):
    predicate_npequal_1 = lambda x: np.equal(x, 1)
    predicate_match = np.logical_and
    reduce_function = np.sum
    return _generic_confusion_matrix_forcat(
        Ytrue,
        predicate_npequal_1,
        Ypred,
        predicate_npequal_1,
        predicate_match,
        reduce_function,
        reduce_axis=axis,
        reduce_keepdims=keepdims)


def confusion_matrix(ytrue,
                     ypred,
                     nclasses=None,
                     axis=None,
                     keepdims=False,
                     warn=False):
    if not isinstance(ytrue, np.ndarray):
        ytrue = np.array(ytrue)

    if not isinstance(ypred, np.ndarray):
        ypred = np.array(ypred)

    # TODO: [ ] Add proper assertions to avoid obvious mistakes

    # assert ytrue.shape == ypred.shape, "Shape mismatch: True {} != {} Predictions".format(
    #     ytrue.shape, ypred.shape)
    # assert len(ytrue.shape) == 1, (
    #     "Only supports vectors of class labels. "
    #     "If your labels are one-hot, please use confusion_matrix_forcategorical"
    # )
    # NOTE: Can't use this, cuz case of multi-pred

    Ytrue = to_categorical(ytrue, nclasses=nclasses, warn=warn)

    # Ytrue tells the correct nclasses now
    Ypred = to_categorical(ypred, nclasses=Ytrue.shape[-1], warn=warn)

    return confusion_matrix_forcategorical(
        Ytrue, Ypred, axis=axis, keepdims=keepdims)


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
