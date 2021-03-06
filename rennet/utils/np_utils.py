#  Copyright 2018 Fraunhofer IAIS. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Numpy utilities

@motjuste
Created: 28-09-2016
"""
from __future__ import division, print_function
from collections import Iterable
from itertools import repeat
from contextlib import contextmanager
import numpy as np
from six.moves import zip, range


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


def strided_view(arr, win_shape, step_shape):  # pylint: disable=too-complex
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
    - if len(win_shape) = d is smaller than len(arr.shape), arr is strided in the first d dimensions
    - if any value at position d is None in win_shape or step_shape, arr is not strided for that dim
        + e.g. win_shape=(None, None, w) and step_shape=(None, None, s) will only stride in 3rd dim.
        + NOTE: if a value at d is None in one, it should also be None in the other at d
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("arr should be a Numpy.ndarray")

    # validate shape of inputs
    if isinstance(win_shape, Iterable) != isinstance(step_shape, Iterable):
        # one is iterable, the other is not
        raise ValueError(
            "Both win_shape: {} and step_shape: {} should be iterable or not iterable.".
            format(win_shape, step_shape)
        )
    elif (
            isinstance(win_shape, Iterable)
            and (len(win_shape) != len(step_shape) or len(win_shape) > len(arr.shape))
    ):  # yapf: disable
        raise ValueError(
            "iterables win_shape: {} and step_shape: {} should be of the same length,"
            "and, the length should be <= len(arr.shape) : {}".format(
                win_shape, step_shape, arr.shape
            )
        )
    elif (
            not isinstance(win_shape, Iterable) and (
                not (isinstance(win_shape, int) and isinstance(step_shape, int)) or
                (win_shape <= 0 or step_shape <= 0)
            )
    ):  # yapf: disable
        # both are not iterable
        raise ValueError(
            "Both win_shape: {} and step_shape: {} should be positive integers.".format(
                win_shape, step_shape
            )
        )

    # create appropriate shaped inputs
    if not isinstance(win_shape, Iterable):
        win_shape = (win_shape, )
        step_shape = (step_shape, )

    nones_toadd = (None, ) * (len(arr.shape) - len(win_shape))
    _win_shape = tuple(win_shape) + nones_toadd
    _step_shape = tuple(step_shape) + nones_toadd

    final_shape = tuple()
    final_strides = tuple()

    for dim, (win, step, shape,
              stride) in enumerate(zip(_win_shape, _step_shape, arr.shape, arr.strides)):
        if win is None != step is None:
            raise ValueError(
                "BOTH win_shape: {} and step_shape: {} should be None at [{}] "
                "or not None.".format(win_shape, step_shape, dim)
            )
        elif win is None:
            # Not striding in this dim
            final_shape += (shape, )
            final_strides += (stride, )
        elif win <= 0 or step <= 0:
            raise ValueError(
                "Both win_shape: {} and step_shape: {} should be > 0 or None "
                "(if you want to skip striding in a dimension) at [{}].".format(
                    win_shape, step_shape, dim
                )
            )
        elif win > shape:
            raise ValueError(
                "win_shape: {} should be <= arr.shape: {} at [{}], given: {} > {}".format(
                    win_shape, arr.shape, dim, win, shape
                )
            )
        else:
            # s is zero when step is larger, resulting in one stride in this dim
            s = (shape - (win - step)) // step
            final_shape += (
                s or 1,
                win,
            )
            final_strides += (
                stride * (step if s else 1),
                stride,
            )

    return np.lib.stride_tricks.as_strided(arr, shape=final_shape, strides=final_strides)


def _apply_rolling(func, arr, win_len, *args, step_len=1, axis=0, **kwargs):
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


def rolling_mean(arr, win_len, *args, axis=0, **kwargs):
    return _apply_rolling(np.mean, arr, win_len, axis=axis, *args, **kwargs)


def rolling_sum(arr, win_len, *args, axis=0, **kwargs):
    return _apply_rolling(np.sum, arr, win_len, axis=axis, *args, **kwargs)


def rolling_max(arr, win_len, *args, axis=0, **kwargs):
    return _apply_rolling(np.max, arr, win_len, axis=axis, *args, **kwargs)


def rolling_min(arr, win_len, *args, axis=0, **kwargs):
    return _apply_rolling(np.min, arr, win_len, axis=axis, *args, **kwargs)


def group_by_values(values):
    # TODO: [A] make it work with 1 dimensional arrays
    # Ref: http://stackoverflow.com/questions/4651683

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
            "Some class labels are greater than provided nclasses: {} > {}".format(
                ymax, nclasses
            )
        )
    elif nclasses > ymax and warn:
        raise RuntimeWarning(
            "Some class labels may be missing: {} > {}".format(nclasses, ymax)
        )

    res = np.arange(nclasses)[np.newaxis, :] == y[..., np.newaxis]
    res = res.astype(np.float)  # cuz it is a probability dist

    return res


def _generic_confmat_forcat(  #pylint: disable=too-many-arguments
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

    _valid_axes = [i for i in range(len(Ypred.shape) - 1) if Ypred.shape[i] > 1]
    if not _valid_axes:
        msg = """ No valid reduction axes found\n
        - Are there more than one examples, at all? Check Shape.\n\tTrue: {}, Pred: {}\n
        - Are you providing categorical (one-hot) labels? Check Values.\n\tTRUE: {}\n\tPRED: {}
        """.format(Ytrue.shape, Ypred.shape, Ytrue, Ypred)
        raise ValueError(msg)
    if reduce_axis is None:
        # choose the last axis > 1, excluding the ClassLabel axis
        # will raise error if none qualify, since then max is looking into empty
        reduce_axis = max(_valid_axes)
    elif reduce_axis not in _valid_axes:
        msg = "The axis argument cannot be:\n"
        msg += "- The last axis (axis of class label) {}\n".format(
            "TRUE" if reduce_axis == len(Ypred.shape) - 1 else ""
        )
        msg += "- An axis of size <= 1 {}".format(
            "TRUE" if Ypred.shape[reduce_axis] <= 1 else ""
        )
        raise ValueError(msg)

    conf = reduce_function(
        predicate_match(
            predicate_true(Ytrue[..., np.newaxis]),
            predicate_pred(Ypred[..., np.newaxis, :])
        ),
        axis=reduce_axis,
        keepdims=reduce_keepdims
    )

    return conf


def confusion_matrix_forcategorical(Ytrue, Ypred, axis=None, keepdims=False):
    predicate_npequal_1 = lambda x: np.equal(x, 1)
    predicate_match = np.logical_and
    reduce_function = np.sum
    return _generic_confmat_forcat(
        Ytrue,
        predicate_npequal_1,
        Ypred,
        predicate_npequal_1,
        predicate_match,
        reduce_function,
        reduce_axis=axis,
        reduce_keepdims=keepdims
    )


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

    return confusion_matrix_forcategorical(Ytrue, Ypred, axis=axis, keepdims=keepdims)


def normalize_confusion_matrix(conf_matrix):
    if not isinstance(conf_matrix, np.ndarray):
        conf_matrix = np.array(conf_matrix)

    conf_sum = conf_matrix.sum(axis=-2)
    confprec = conf_matrix / conf_sum[..., np.newaxis, :]

    conf_sum = conf_matrix.sum(axis=-1)
    confrec = conf_matrix / conf_sum[..., :, np.newaxis]

    return confprec, confrec


@contextmanager
def printoptions(*args, **kwargs):
    orig_options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**orig_options)


def print_prec_rec(prec, rec, onlydiag=False, end='\n'):
    p = prec * 100
    r = rec * 100
    if onlydiag:
        print(
            "P(REC)){}{}".format(
                "{:^2}".format(' '), "{:^2}".format(' ').join(
                    "{: >6.2f} ({: >6.2f})".format(*z)
                    for z in zip(p.diagonal(), r.diagonal())
                )
            ),
            end=end
        )
    else:
        n = prec.shape[-1]
        tpf = "".join(["{:^", str(n * 7 + 2), "}"]).format("PRECISION")
        trf = "".join(["{:^", str(n * 7 + 1), "}"]).format("RECALL")
        spc = "{:^6}".format(' ')
        print("".join([tpf, spc, trf]))

        with printoptions(suppress=True, formatter={'float': '{: >6.2f}'.format}):
            print(
                "\n".join("{}{}{}".format(*z) for z in zip(p, repeat(spc), r)),
                end="\n" + end if end != "\n" else end
            )


def normalize_dynamic_range(arr, new_min=0., new_max=1., axis=None):
    if np.ndim(new_min) != 0 or np.ndim(new_max) != 0:
        raise NotImplementedError("Only implemented for scalar new_min and new_max")

    # we do min-/max-ing without NaN values causing issues
    amin = arr.nanmin(axis=axis)
    amax = arr.nanmax(axis=axis)

    if axis is not None:
        amin = np.expand_dims(amin, axis=axis)
        amax = np.expand_dims(amax, axis=axis)

    return new_min + (arr - amin) * (new_max - new_min) / (amax - amin)


def normalize_mean_std_rolling(  # pylint: disable=too-complex
        arr, win_len, *args, axis=0, std_it=True, first_mean_var='skip', **kwargs
): # yapf: disable
    """ Mean-Variance normalize a given numpy.ndarray in a rolling way.

    NOTE: `first_mean_var` decides what to use for the first `win_len` elements of `arr`.
    - 'skip' : don't normalize
    - 'copy' : copy the first mean and std values along the given `axis` and use those
    - `tuple(mean, std)` : use the values from the tuple

    When the params are none of these, or the tuple's size != 2, a `ValueError` is raised.
    """
    if first_mean_var not in ['skip', 'copy'] and not isinstance(first_mean_var, tuple):
        raise ValueError(
            "first_mean_var should be either : 'skip', 'copy' or a tuple(mean, std) of length 2"
        )
    elif isinstance(first_mean_var, tuple):
        if len(first_mean_var) != 2:
            raise ValueError(
                "first_mean_var should be either : 'skip', 'copy' or a tuple(mean, std) of length 2"
            )
        elif any(
                (len(mv.shape) == len(arr.shape)) and
                all((m == a) or (i == axis)
                    for i, (m, a) in enumerate(zip(mv.shape, arr.shape)))
                for mv in first_mean_var
        ):  # yapf: disable
            raise ValueError(
                "Mismatch in shapes of provided first_mean_var and arr.\n" +
                "The shape should at least be 1 along given `axis`"
            )
    elif axis < 0 or axis + 1 > len(arr.shape):
        raise ValueError("axis should be >= 0 and within the shape of the given array")
    elif win_len < 2:
        raise ValueError("Such small win_len is not supported, and perhaps unnecessary")

    rmean = _apply_rolling(np.mean, arr, win_len, axis=axis, *args, **kwargs)
    if std_it:
        rstd = _apply_rolling(np.std, arr, win_len, axis=axis, *args, **kwargs)
    else:
        rstd = 1

    if first_mean_var == 'skip':
        arridx = (slice(0, None, 1), ) * axis + (
            slice(win_len - 1, None, 1),
            Ellipsis,
        )
        return (arr[arridx] - rmean) / rstd

    if first_mean_var == 'copy':
        ridx = (slice(0, None, 1), ) * axis + (
            slice(0, 1, 1),
            Ellipsis,
        )
        first_mean_std = (rmean[ridx], rstd[ridx] if std_it else rstd)
    else:  # first_mean_var has been given and is of the right shape
        #                                     variance to std
        first_mean_std = first_mean_var[:1] + np.sqrt(first_mean_var[-1])

    rmean = np.insert(rmean, slice(0, win_len - 1, 1), first_mean_std[0], axis=axis)
    if std_it:
        rstd = np.insert(rstd, slice(0, win_len - 1, 1), first_mean_std[1], axis=axis)

    return (arr - rmean) / rstd
