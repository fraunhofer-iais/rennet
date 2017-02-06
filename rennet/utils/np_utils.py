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


def to_categorical(y, nclasses=None):
    """ Convert class vectors to one-hot class matrix

    # Parameters
        y: 1D numpy class vector to be converted.
        nclasses: optional total number of classes

    # Returns
        A one-hot encodede class matrix

    # Raises
        RuntimeError: when any y is greater than nclasses
    """
    ymax = np.max(y)
    if nclasses is None:
        nclasses = ymax
    elif nclasses < ymax:
        raise RuntimeError(
            "Some class labels are greater than provided nclasses: {} > {}".
            format(ymax, nclasses))

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
