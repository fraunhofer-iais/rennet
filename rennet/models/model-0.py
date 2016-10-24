"""
@motjuste
Creadted: 24-10-2016

First 3 layer model for DoubleTalk to test functionality
"""
from __future__ import division, print_function, absolute_import
import os
import sys
import numpy as np
import h5py
import glob
from scipy.signal import medfilt
import keras.layers as kl
import keras.optimizers as ko
from keras.models import Sequential
from keras.utils import np_utils

RENNET_ROOT = os.environ["RENNET_ROOT"]
sys.path.join(RENNET_ROOT)


def picklepaths(workingdir, project, dataset, wildcard):
    d = os.path.join(workingdir, project, dataset)
    # TODO: Check for existence of the directory

    return glob.glob(os.path.join(d, wildcard))


def read_pickles(filepaths, concatenate=True, keys=('data', 'labels')):
    data = dict.fromkeys(keys)

    # TODO: Check for existence of file
    for f in filepaths:
        d = h5py.File(f, 'r')
        for k in data:
            data[k] = d[k][()]

    if concatenate:
        for k in data:
            data[k] = np.concatenate(data[k])

    return data


def gertvdata():
    working_dir = os.path.join(RENNET_ROOT, "data", "working")

    proj = 'gertv1000-utt'
    dataset = 'AudioMining'

    trn_wildcard = os.path.join('train', 'pickles', '20161019*.hdf5')

    trn_fps = picklepaths(working_dir, proj, dataset, trn_wildcard)
    if len(trn_fps) == 0:
        raise RuntimeError("No Training file found at:\
                {}".format(
            os.path.join(working_dir, proj, dataset, trn_wildcard)))

    d = read_pickles(trn_fps)
    data = d['data']
    labels = d['labels']

    assert data.shape[0] == labels.shape[0], "data and label shapes \
        mismatch: {} v/s {}".format(data.shape, labels.shape)

    return data, labels
