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

        raise NotImplementedError("Not implemented in base class {}".format(
            self.__class__.__name__))

    @property
    def chunkings(self):
        """ Chunking information for each dataset.

        List of Chunking instances, with slices corresponding to the each chunk
        of the datasets that will be read.

        It is a computed property, because these are read only when
        we have the final list of all the datasets to be read from the file.
        """

        raise NotImplementedError("Not implemented in base class {}".format(
            self.__class__.__name__))


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
        raise NotImplementedError("Not implemented in base class {}".format(
            self.__class__.__name__))

    def prep_label(self, label):
        """ Do anything additional to the read label like normalize, reshape, etc.

        Here to enforce API.
         """
        raise NotImplementedError("Not implemented in base class {}".format(
            self.__class__.__name__))

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
        elif isinstance(shuffle_seed, int):
            nr.random.seed(shuffle_seed)
            return nr.random.shuffle(arr)
        else:
            raise ValueError(
                "shuffle_seed should either be None (no shuffling) or an integer"
            )

    def get_prepped_inputs(self, chunking, shuffle_seed=None, **kwargs):  # pylint: disable=unused-argument
        """ The method that will be called by DataProvider (below).

        Override if, for example, you want to also have sample_weights returned.
        """
        data, label = self.get_prepped_data_label(chunking)

        # NOTE: since the same shufflng seed is used for all arrays,
        # The correspondence will be preserved
        return (self.maybe_shuffle(data, shuffle_seed),
                self.maybe_shuffle(label, shuffle_seed))


class BaseInputsProvider(BaseH5ChunkingsReader, BaseH5ChunkPrepper):  # pylint: disable=abstract-method
    def __init__(self,
                 filepath,
                 shuffle_seed=None,
                 nepochs=1,
                 batchsize=None,  # None => chunksize
                 **kwargs):
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

    def reset(self):
        nchunks = len(self.chunkings)
        self.setup_shuffling_seeds(nchunks)

    def flow(self):

        self.reset()

        nchunks = len(self.chunkings)

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

                data_label = self.get_prepped_inputs(
                    chunking, shuffle_seed=chunking_seed)

                yield data_label

"""
fit_generator(self, generator,
steps_per_epoch,  TOTAL_LENGTH in an epoch
epochs=1,
verbose=1,
callbacks=None,
validation_data=None, validation_steps=None,
class_weight=None,
max_q_size=10,
workers=1, pickle_safe=False, initial_epoch=0)
"""
