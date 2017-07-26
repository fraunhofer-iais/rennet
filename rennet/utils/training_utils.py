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
import time

from rennet.utils.py_utils import threadsafe_generator


""" Basic namedtuple for Chunking information.
But it is perfectly acceptable to just pass a custom Chunking class.
Just meet the following expectations:
- One chunking is intended to be read as :
    "The data in this dataslice at this datapath has it's corresponding
     label at the labelslice at this labelpath"
- The slices should be valid slice or np.s_ instances.
"""
Chunking = namedtuple('Chunking', [
    'datapath',
    'dataslice',
    'labelpath',
    'labelslice',
])


class BaseH5ChunkingsReader(object):
    """ Base class for reading data and labels chunking info from an HDF5 file.

    The purpose of this class to setup an expected API by other derived classes,
    and to carry out some of the computations that are actually not specilized.
    """

    def __init__(self, filepath, **kwargs):  # pylint: disable=unused-argument
        self.filepath = filepath

    @property
    def totlen(self):
        """ Total length of the entire dataset that will be read

        It is a computed property, because these are read only when
        we have the final list of all the datasets to be read from the file.
        """

        raise NotImplementedError(
            "Not implemented in base class {}".format(self.__class__.__name__))

    @property
    def chunkings(self):
        """ Chunking information for each dataset.

        List of Chunking instances, with slices corresponding to the each chunk
        of the datasets that will be read.

        It is a computed property, because these are read only when
        we have the final list of all the datasets to be read from the file.
        """

        raise NotImplementedError(
            "Not implemented in base class {}".format(self.__class__.__name__))

    @property
    def nchunks(self):
        """ Total number of chunks for which chunking information is available.
        """
        return len(self.chunkings)


class BaseH5ChunkPrepper(object):
    """ Base class for reading and prepping data and labels from an HDF5 file chunkwise.
    It also implements shuffling of the data when a valid seed is provided.

    The purpose of this class to setup an expected API by other derived classes,
    and to carry out some of the computations that are actually not specilized.
    """

    def __init__(self, filepath, **kwargs):  # pylint: disable=unused-argument
        self.filepath = filepath

    def read_h5_data_label_chunk(self, chunking):
        """ Read the data and label chunks from the HDF5 file. """

        with h.File(self.filepath, 'r') as f:
            data = f[chunking.datapath][chunking.dataslice]
            label = f[chunking.labelpath][chunking.labelslice]

        return data, label

    def prep_data(self, data):
        """ Do anything additional to the read data like normalize, reshape, etc.

        Here to enforce API.
         """
        raise NotImplementedError(
            "Not implemented in base class {}".format(self.__class__.__name__))

    def prep_label(self, label):
        """ Do anything additional to the read label like normalize, reshape, etc.

        Here to enforce API.
         """
        raise NotImplementedError(
            "Not implemented in base class {}".format(self.__class__.__name__))

    def get_prepped_data_label(self, chunking):
        """ Get the prepped data and label chunks.

        Override the prep_data and prep_label methods to play with the data
        before it is provided as inputs.
        """
        data, label = self.read_h5_data_label_chunk(chunking)

        return self.prep_data(data), self.prep_label(label)

    @classmethod
    def maybe_shuffle(self, arr, shuffle_seed):
        if shuffle_seed is None:
            return arr
        elif isinstance(shuffle_seed, (int, np.int_)):
            nr.seed(shuffle_seed)
            nr.shuffle(arr)
            return arr
        else:
            raise ValueError(
                "shuffle_seed should either be None (no shuffling) or an integer"
                " {} was given {}".format(shuffle_seed, type(shuffle_seed)))

    def get_prepped_inputs(self, chunking, shuffle_seed=None, **kwargs):  # pylint: disable=unused-argument
        """ The method that will be called by DataProvider (below).

        Override if, for example, you want to also have sample_weights returned.
        However, other classes expect the first two arrays in the tuple to be
        data and label, in that order.
        """
        data, label = self.get_prepped_data_label(chunking)

        # NOTE: since the same shufflng seed is used for all arrays,
        # The correspondence will be preserved
        return (self.maybe_shuffle(data, shuffle_seed), self.maybe_shuffle(
            label, shuffle_seed))


class BaseInputsProvider(BaseH5ChunkingsReader, BaseH5ChunkPrepper):  # pylint: disable=abstract-method
    def __init__(self, filepath, shuffle_seed=None, nepochs=1, **kwargs):
        if shuffle_seed is not None and not isinstance(shuffle_seed, int):
            raise ValueError(
                "shuffle_seed should be either None (no shuffling) or an integer, Found: {}".
                format(shuffle_seed))
        else:
            self.shuffle_seed = shuffle_seed

        super(BaseInputsProvider, self).__init__(filepath, **kwargs)

        if nepochs <= 0:
            raise ValueError("nepochs should be >= 1")

        self.nepochs = nepochs
        self._input_shape = None

        self.epoch_shuffle_seeds = None  # seeds to permute chunk indices, one per epoch
        self.chunk_shuffle_seeds = None  # seeds to permute sample indices, one per chunk per epoch

    @property
    def input_shape(self, **kwargs):
        if self._input_shape is None:
            data = self.get_prepped_inputs(self.chunkings[0], **kwargs)[0]
            self._input_shape = ((None, ) + data.shape[1:])

        return self._input_shape

    def setup_shuffling_seeds(self):
        if not self.shuffle_seed is None:
            nseedsrequired = self.nepochs + (self.nchunks * self.nepochs)
            nr.seed(self.shuffle_seed)
            seeds = nr.randint(41184535, size=nseedsrequired)

            self.epoch_shuffle_seeds = seeds[:self.nepochs]
            if self.nepochs == 1:
                self.epoch_shuffle_seeds = [self.epoch_shuffle_seeds]

            self.chunk_shuffle_seeds = seeds[self.nepochs:]
            if not isinstance(self.chunk_shuffle_seeds, np.ndarray):
                # Single epoch, single chunk
                self.chunk_shuffle_seeds = np.array(
                    [[self.chunk_shuffle_seeds]])
            else:
                self.chunk_shuffle_seeds = np.reshape(
                    self.chunk_shuffle_seeds, (self.nepochs, self.nchunks))

    @threadsafe_generator
    def flow(self, indefinitely=False, sleepsec_after_epoch=0):
        self.setup_shuffling_seeds()
        for e in range(self.nepochs):
            for inputs in self.flow_for_epoch_at(e):
                yield inputs
            time.sleep(sleepsec_after_epoch)

        while indefinitely:  # keras expects indefinitely running generator
            for e in range(self.nepochs):
                for inputs in self.flow_for_epoch_at(e):
                    yield inputs
                time.sleep(sleepsec_after_epoch)

    def flow_for_epoch_at(self, at):
        # setup chunks order
        if self.epoch_shuffle_seeds is not None:
            nr.seed(self.epoch_shuffle_seeds[at])
            chunk_idx = nr.permutation(self.nchunks)
        else:
            chunk_idx = np.arange(self.nchunks)

        if self.chunk_shuffle_seeds is not None:
            chunking_seeds = self.chunk_shuffle_seeds[at]
        else:
            chunking_seeds = [None] * self.nchunks
        # Start yielding chunks
        for c in range(self.nchunks):
            chunking = self.chunkings[chunk_idx[c]]
            seed = chunking_seeds[c]

            inputs = self.get_prepped_inputs(chunking, shuffle_seed=seed)

            yield inputs
