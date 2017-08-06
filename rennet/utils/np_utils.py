"""
@motjuste
Created: 28-09-2016

Numpy utilities
"""
from __future__ import division, print_function
import numpy as np
from collections import Iterable
from itertools import repeat
from contextlib import contextmanager


def base_array_of(arr):
    """ get the base numpy array that owns the data in `arr` """
    base = arr
    while isinstance(base.base, (np.ndarray, np.lib.stride_tricks.DummyArray)):
        base = base.base
    return base


def arrays_do_share_data(arr1, arr2):
    """ Find if the two numpy arrays share the same base data

    Ref
        - https://github.com/ipython-books/cookbook-code/issues/2
    """
    return base_array_of(arr1) is base_array_of(arr2)


def totuples(arr):
    """ Convert a `numpy.ndarray` to tuple of tuples of tuples of ...

    Complete HACK to mimic `numpy.ndarray.tolist`, but instead to get hashable
    `tuple(tuple(tuple(...)))`.

    Reference
        - https://stackoverflow.com/a/10016613
    """
    try:
        return tuple(totuples(_a) for _a in arr)
    except TypeError:
        return arr


def strided_view(arr, win_shape, step_shape):
    """ Create strided view of `arr` without copying

    NOTE: There are some restrictions to use of this method, considering numpy
    suggests not using the underlying striding trick as far as possible.

    It is suggested that this function be applied to copies of arrays, as later
    operations may result in overwriting of the underlying data. This is
    because the result from this function is just rearranged references to the
    underlying data in arr, and any in-place changes may corrupt arr itself.

    `win_shape` and `step_shape` can be positive integers > 0, or
    iterables (of the same size) of positive integers > 0 or None.

    When they are integers:
    - the view is created by striding only in the first dimension of arr
    - `win_shape` should be at most equal to len(arr)
    - `step_shape` > `win_shape` will result in skipping of sub-arrays
        + however, the first sub-array for a `win_shape` is always present
    - `step_shape` > len(arr) will result in only a single stride
        + essentially of shape starting as (1, win_shape, ...)

    When they are iterables:
    - len(win_shape) should be equal to len(step_shape)
    - len(win_shape) should be atmost len(arr.shape)
    - when len(win_shape) = d is smaller than len(arr.shape), arr is strided in the first d dimensions
    - when any value at position d, either in win_shape or step_shape is None, arr is not strided for that dimension
        + For striding in only one dimension, other than the first, pass iterables with None for all dimension before it
        + e.g. win_shape=(None, None, w) and step_shape=(None, None, s) to stride in the third dimension only,
        + NOTE: if a value at d is None in one, it should also be None in the other at d
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("arr should be a Numpy.ndarray")

    # validate shape of inputs
    if isinstance(win_shape, Iterable) != isinstance(step_shape, Iterable):
        # one is iterable, the other is not
        raise ValueError(
            "Both win_shape: {} and step_shape: {} should be iterable or not iterable.".
            format(win_shape, step_shape))
    elif (isinstance(win_shape, Iterable) and
          (len(win_shape) != len(step_shape) or
           len(win_shape) > len(arr.shape))):
        raise ValueError(
            "iterables win_shape: {} and step_shape: {} should be of the same length,"
            "and, the length should be <= len(arr.shape) : {}".format(
                win_shape, step_shape, arr.shape))
    else:
        # both are not iterable
        if (not isinstance(win_shape, Iterable) and
            (not (isinstance(win_shape, int) and isinstance(step_shape, int))
             or (win_shape <= 0 or step_shape <= 0))):
            raise ValueError(
                "Both win_shape: {} and step_shape: {} should be positive integers.".
                format(win_shape, step_shape))

    # create appropriate shaped inputs
    if not isinstance(win_shape, Iterable):
        win_shape = (win_shape, )
        step_shape = (step_shape, )

    nones_toadd = (None, ) * (len(arr.shape) - len(win_shape))
    _win_shape = tuple(win_shape) + nones_toadd
    _step_shape = tuple(step_shape) + nones_toadd

    final_shape = tuple()
    final_strides = tuple()

    for d, (
            win, step, shape, stride
    ) in enumerate(zip(_win_shape, _step_shape, arr.shape, arr.strides)):
        if win is None != step is None:
            raise ValueError(
                "BOTH win_shape: {} and step_shape: {} should be None at [{}] "
                "or not None.".format(win_shape, step_shape, d))
        elif win is None:
            # Not striding in this dim
            final_shape += (shape, )
            final_strides += (stride, )
        elif win <= 0 or step <= 0:
            raise ValueError(
                "Both win_shape: {} and step_shape: {} should be > 0 or None "
                "(if you want to skip striding in a dimension) at [{}].".format(
                    win_shape, step_shape, d))
        elif win > shape:
            raise ValueError(
                "win_shape: {} should be <= arr.shape: {} at [{}], given: {} > {}".
                format(win_shape, arr.shape, d, win, shape))
        else:
            # s is zero when step is larger, resulting in one stride in this dim
            s = (shape - (win - step)) // step
            final_shape += (s or 1, win, )
            final_strides += (stride * (step if s else 1), stride, )

    return np.lib.stride_tricks.as_strided(
        arr, shape=final_shape, strides=final_strides)


def _apply_rolling(func, arr, win_len, step_len=1, axis=0, *args, **kwargs):
    """ Apply a numpy function (that supports acting across an axis) in a rolling way

    That is, apply `func` to `win_len` number of items along `axis`, every
    `step_len`. This is equivalent in functionality as the code below, but
    much more efficient and quicker.

    ```
    # example for applying func along first axis
    result = []
    for start in range(0, len(arr), step_len):
        result.append(func(arr[start:start+win_len, ...],
            axis=0, *args, **kwargs))
    result = np.array(result)
    ```

    Check https://docs.scipy.org/doc/numpy/reference/routines.statistics.html
    and some sample rolling functions below for inspiration.

    NOTE: `func` must support application along an axis

    TODO: [ ] Dox
    """
    win_shape = [None] * len(arr.shape)
    win_shape[axis] = win_len
    step_shape = [None] * len(arr.shape)
    step_shape[axis] = step_len
    strided_arr = strided_view(arr, win_shape, step_shape)

    return func(strided_arr, axis=axis + 1, *args, **kwargs)


def rolling_mean(arr, win_len, axis=0, *args, **kwargs):
    return _apply_rolling(np.mean, arr, win_len, axis=axis, *args, **kwargs)


def rolling_sum(arr, win_len, axis=0, *args, **kwargs):
    return _apply_rolling(np.sum, arr, win_len, axis=axis, *args, **kwargs)


def rolling_max(arr, win_len, axis=0, *args, **kwargs):
    return _apply_rolling(np.max, arr, win_len, axis=axis, *args, **kwargs)


def rolling_min(arr, win_len, axis=0, *args, **kwargs):
    return _apply_rolling(np.min, arr, win_len, axis=axis, *args, **kwargs)


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


def _generic_confusion_matrix_forcat(  #pylint: disable=too-many-arguments
        Ytrue,
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
    if len(_valid_axes) == 0:
        msg = """ No valid reduction axes found\n
        - Are there more than one examples, at all? Check Shape.\n\tTrue: {}, Pred: {}\n
        - Are you providing categorical (one-hot) labels? Check Values.\n\tTRUE: {}\n\tPRED: {}
        """.format(Ytrue.shape, Ypred.shape, Ytrue, Ypred)
        raise ValueError(msg)
    if reduce_axis is None:
        # choose the last axis > 1, excluding the ClassLabel axis
        # will raise error if none qualify, since then max is looking into empty
        reduce_axis = max(_valid_axes)
    else:
        if reduce_axis not in _valid_axes:
            msg = """The axis argument cannot be:
            - The last axis (axis of class label) {}
            - An axis of size <= 1 {}
            """.format(
                "TRUE" if reduce_axis == len(Ypred.shape) - 1 else "",
                "TRUE" if Ypred.shape[reduce_axis] <= 1 else "", )
            raise ValueError(msg)

    conf = reduce_function(
        predicate_match(
            predicate_true(Ytrue[..., np.newaxis]),
            predicate_pred(Ypred[..., np.newaxis, :])),
        axis=reduce_axis,
        keepdims=reduce_keepdims)

    return conf


def confusion_matrix_forcategorical(Ytrue, Ypred, axis=None, keepdims=False):
    predicate_npequal_1 = lambda x: np.equal(x, 1)  #pylint: disable=no-member
    predicate_match = np.logical_and  #pylint: disable=no-member
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


def confusion_matrix(  #pylint: disable=too-many-arguments
        ytrue,
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


def normalize_confusion_matrices(conf_matrix):
    if not isinstance(conf_matrix, np.ndarray):
        conf_matrix = np.array(conf_matrix)

    conf_sum = conf_matrix.sum(axis=-2)
    confprec = conf_matrix / conf_sum[..., np.newaxis, :]

    conf_sum = conf_matrix.sum(axis=-1)
    confrec = conf_matrix / conf_sum[..., :, np.newaxis]

    return confprec, confrec


normalize_confusion_matrix = normalize_confusion_matrices


@contextmanager
def printoptions(*args, **kwargs):
    og = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**og)


def print_normalized_confusion(confmat, title='CONFUSION MATRIX'):
    print("\n{:/>70}//".format(" {} ".format(title)))
    with printoptions(suppress=True, formatter={'float': '{: >6.2f}'.format}):
        print(confmat * 100)


def print_prec_rec(prec, rec, onlydiag=False):
    p = prec * 100
    r = rec * 100
    if onlydiag:
        print("P(REC)){}{}".format("{:^2}".format(' '), "{:^2}".format(
            ' ').join("{: >6.2f} ({: >6.2f})".format(*z)
                      for z in zip(p.diagonal(), r.diagonal()))))
    else:
        n = prec.shape[-1]
        tpf = "".join(["{:^", str(n * 7 + 2), "}"]).format("PRECISION")
        trf = "".join(["{:^", str(n * 7 + 1), "}"]).format("RECALL")
        spc = "{:^6}".format(' ')
        print("".join([tpf, spc, trf]))

        with printoptions(
                suppress=True, formatter={'float': '{: >6.2f}'.format}):
            print(
                "\n".join("{}{}{}".format(*z) for z in zip(p, repeat(spc), r)))
