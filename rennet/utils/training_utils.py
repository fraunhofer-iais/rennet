"""
@motjuste
Created: Mon, 10-Apr-2017

Utilities for training
"""
from __future__ import print_function, division
from collections import namedtuple
import numpy as np
import numpy.random as nr
import h5py as h


class BaseH5ChunkingsReader(object):

    Chunking = namedtuple('Chunking', [
        'datapath',
        'dataslice',
        'labelpath',
        'labelslice',
    ])

    def __init__(self, filepath, **kwargs):  # pylint: disable=unused-argument
        self.filepath = filepath

    @property
    def totlen(self):
        raise NotImplementedError("Not implemented in Base class")

    @property
    def chunkings(self):
        raise NotImplementedError("Not implemented in Base class")


class BaseDataPrepper(object):
    def __init__(self, filepath, **kwargs):  # pylint: disable=unused-argument
        self.filepath = filepath

    def read_h5_data_label(self, datapath, dataslice, labelpath, labelslice):
        with h.File(self.filepath, 'r') as f:
            data = f[datapath][dataslice]
            label = f[labelpath][labelslice]

        return data, label

    def prep_data(self, data):
        raise NotImplementedError("Not implemented in base class")

    def prep_label(self, label):
        raise NotImplementedError("Not implemented in base class")

    def get_prepped_data_label(  # pylint: disable=too-many-arguments
            self,
            datapath,
            dataslice,
            labelpath,
            labelslice,
            shuffle_seed=None):
        data, label = self.read_h5_data_label(datapath, dataslice, labelpath,
                                              labelslice)
        data = self.prep_data(data)
        label = self.prep_label(label)

        if shuffle_seed is None:
            return data, label
        elif isinstance(shuffle_seed, int):
            nr.random.seed(shuffle_seed)
            nr.random.shuffle(data)
            nr.random.seed(shuffle_seed)
            nr.random.shuffle(label)
        else:
            raise ValueError(
                "shuffle_seed should be either None (no shuffling) or an integer"
            )


class BaseDataProvider(BaseH5ChunkingsReader, BaseDataPrepper):  # pylint: disable=abstract-method
    def __init__(self,
                 filepath,
                 shuffle_seed=None,
                 nepochs=1,
                 batchsize=None,
                 **kwargs):
        if shuffle_seed is not None and not isinstance(shuffle_seed, int):
            raise ValueError(
                "shuffle_seed should be either None (no shuffling) or an integer, Found: {}".
                format(shuffle_seed))
        else:
            self.shuffle_seed = shuffle_seed

        super(BaseDataProvider, self).__init__(filepath, **kwargs)

        if nepochs <= 0:
            raise ValueError("nepochs should be >= 1")
        self.nepochs = nepochs
        self.batchsize = batchsize

        self.epoch_shuffle_seeds = None  # seeds to permute chunk indices, one per epoch
        self.chunk_shuffle_seeds = None  # seeds to permute sample indices, one per chunk per epoch

    def setup_shuffling_seeds(self, nchunks):
        if not self.shuffle_seed is None:
            nr.random.seed(self.shuffle_seed)
            nseedsrequired = self.nepochs + (nchunks * self.nepochs)
            seeds = nr.random.randint(size=nseedsrequired)

            self.epoch_shuffle_seeds = seeds[:self.nepochs]
            if self.nepochs == 1:
                self.epoch_shuffle_seeds = [self.epoch_shuffle_seeds]

            self.chunk_shuffle_seeds = seeds[self.nepochs:]
            if not isinstance(self.chunk_shuffle_seeds, np.ndarray):
                # Single epoch, single chunk
                self.chunk_shuffle_seeds = np.array(
                    [[self.chunk_shuffle_seeds]])
            else:
                self.chunk_shuffle_seeds = np.reshape(self.chunk_shuffle_seeds,
                                                      (self.nepochs, nchunks))

    def flow(self):
        # FIXME: Can't call in init, cuz chunkings is a calculated property
        nchunks = len(self.chunkings)
        self.setup_shuffling_seeds(nchunks)

        for e in range(self.nepochs):

            # setup chunks order
            if self.epoch_shuffle_seeds is not None:
                chunk_idx = nr.random.permutation(nchunks)
            else:
                chunk_idx = np.arange(nchunks)

            chunking_seeds = self.chunk_shuffle_seeds[
                e] if self.chunk_shuffle_seeds is not None else [
                    None for _ in range(nchunks)
                ]
            for c in range(nchunks):
                chunking = self.chunkings[chunk_idx[c]]
                chunking_seed = chunking_seeds[c]

                datapath, dataslice = chunking.datapath, chunking.dataslice
                labelpath, labelslice = chunking.labelpath, chunking.labelslice
                (data_label) = self.get_prepped_data_label(
                    datapath,
                    dataslice,
                    labelpath,
                    labelslice,
                    shuffle_seed=chunking_seed)

                yield data_label
