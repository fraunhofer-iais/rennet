"""
@motjuste
Created: Mon, 10-Apr-2017

Utilities for training
"""
from __future__ import print_function, division
from six.moves import zip
from collections import namedtuple, Iterable
from warnings import warn as warning
import numpy as np
import numpy.random as nr
import h5py as h

import rennet.utils.np_utils as nu

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
        """ Total length of the entire dataset.

        NOTE: It depends on the Prepper how many of these data will be read.

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

    @property
    def nchunks(self):
        """ Total number of chunks for which chunking information is available.
        """
        return len(self.chunkings)


# PREPPERS ######################################################### PREPPERS #


class BaseH5ChunkPrepper(object):
    """ Base class for reading and prepping data and labels from an HDF5 file chunkwise.
    It also implements shuffling of the data when a valid seed is provided.

    The purpose of this class to setup an expected API by other derived classes,
    and to carry out some of the computations that are actually not specilized.
    """

    def __init__(self, filepath, **kwargs):  # pylint: disable=unused-argument
        self.filepath = filepath

    def read_h5_data_label_chunk(self, chunking, only_labels=False, **kwargs):  # pylint: disable=unused-argument
        """ Read the data and label chunks from the HDF5 file. """

        with h.File(self.filepath, 'r') as f:
            label = f[chunking.labelpath][chunking.labelslice]
            if not only_labels:
                data = f[chunking.datapath][chunking.dataslice]
            else:
                data = np.empty_like(label)

        return data, label

    def prep_data(self, data, only_labels=False, **kwargs):  # pylint: disable=unused-argument
        """ Do anything additional to the read data like normalize, reshape, etc.

        Here to enforce API.
         """
        if only_labels:
            return data  # This is dummy data
        else:
            raise NotImplementedError("Not implemented in class {}".format(
                self.__class__.__name__))

    def prep_label(self, label, **kwargs):
        """ Do anything additional to the read label like normalize, reshape, etc.

        Here to enforce API.
         """
        raise NotImplementedError("Not implemented in class {}".format(
            self.__class__.__name__))

    def get_prepped_data_label(self, chunking, only_labels=False, **kwargs):
        """ Get the prepped data and label chunks.

        Override the prep_data and prep_label methods to play with the data
        before it is provided as inputs.
        """
        data, label = self.read_h5_data_label_chunk(
            chunking, only_labels=only_labels, **kwargs)

        return self.prep_data(
            data, only_labels=only_labels, **kwargs), self.prep_label(
                label, **kwargs)


class AsIsChunkPrepper(BaseH5ChunkPrepper):
    def prep_data(self, data, only_labels=False, **kwargs):
        return data

    def prep_label(self, label, **kwargs):
        return label


def dominant_label_for_subcontext(labels_in_subcontext):
    # nclasses = labels_in_subcontext.shape[-1]
    # nperclass_subcontext = labels_in_subcontext.sum(axis=axis)
    # dominant_subcontext = nperclass_subcontext.argmax(-1)  # last axis is that of class
    # categorical = nu.to_categorical(dominant_subcontext, nclasses=nclasses)
    # return categorical
    return nu.to_categorical(
        labels_in_subcontext.sum(axis=-2).argmax(axis=-1),
        nclasses=labels_in_subcontext.shape[-1])


def max_label_for_subcontext(labels_in_subcontext):
    # nclasses = labels_in_subcontext.shape[-1]
    # labels_subcontext_noncat = labels_in_subcontext.argmax(axis=-1)
    # max_label_subcontext_noncat = labels_subcontext_noncat.max(axis=-1)
    # categorical = nu.to_categorical(max_label_for_subcontext, nclasses=nclasses)
    # return categorical
    return nu.to_categorical(
        labels_in_subcontext.argmax(axis=-1).max(axis=-1),
        nclasses=labels_in_subcontext.shape[-1])


class BaseWithContextPrepper(BaseH5ChunkPrepper):  # pylint: disable=abstract-method
    def __init__(  # pylint: disable=too-many-arguments
            self,
            filepath,
            data_context=0,
            label_subcontext=0,  # 0 for choosing only the center label as label
            label_from_subcontext_fn=dominant_label_for_subcontext,
            add_channel_at_end=True,
            **kwargs):

        assert data_context >= 0, (
            "data_context should be >= 0, v/s {}".format(data_context))
        self.dctx = data_context
        self.win = 1 + 2 * self.dctx
        self.stp = 1

        assert 0 <= label_subcontext <= data_context, (
            "labels_context should be >= 0, and <= data_context "
            "({}), v/s {}".format(self.dctx, label_subcontext))
        self.lctx = label_subcontext
        self.lctxfn = label_from_subcontext_fn

        self.add_channel = add_channel_at_end

        sup = super(BaseWithContextPrepper, self)
        sup.__init__(filepath, **kwargs)

    def get_prepped_data_label(self, chunking, only_labels=False, **kwargs):
        sup = super(BaseWithContextPrepper, self)
        data, label = sup.get_prepped_data_label(
            chunking, only_labels=only_labels, **kwargs)

        if self.dctx > 0:
            # no context adding for either data or label
            if not only_labels:
                data = nu.strided_view(
                    data, win_shape=self.win, step_shape=self.stp)

            label = nu.strided_view(
                label, win_shape=self.win, step_shape=self.stp)

            if self.lctx == 0:  # only the center frame's label
                label = label[:, self.dctx:self.dctx + 1, ...]
            else:
                label = label[:, self.dctx - self.lctx:
                              self.dctx + self.lctx + 1, ...]
        else:
            label = label[:, np.newaxis, ...]

        if self.add_channel:
            return data[..., None], self.lctxfn(label)
        else:
            return data, self.lctxfn(label)


# NORMALIZERS ################################################### NORMALIZERS #


class BaseDataNormalizer(BaseH5ChunkPrepper):
    def normalize_data(self, data, **kwargs):
        raise NotImplementedError("Not implemented in class {}".format(
            self.__class__.__name__))

    def prep_data(self, data, only_labels=False, **kwargs):
        if only_labels:
            return data  # This is dummy data
        else:
            return self.normalize_data(data, **kwargs)


class BaseChunkMeanVarianceNormalizer(BaseDataNormalizer):  # pylint: disable=abstract-method
    def __init__(self, filepath, mean_it=True, std_it=False, **k):
        self.mean_it = mean_it
        self.std_it = std_it
        super(BaseChunkMeanVarianceNormalizer, self).__init__(filepath, **k)

    def normalize_data(self, data, **kwargs):
        if self.mean_it:
            data = data - data.mean(axis=0)

        if self.std_it:
            data = data / data.std(axis=0)

        return data


# INPUTS PROVIDERS ######################################### INPUTS PROVIDERS #


class BaseInputsProvider(BaseH5ChunkingsReader, BaseH5ChunkPrepper):  # pylint: disable=abstract-method
    def __init__(self, filepath, shuffle_seed=None, npasses=1, **kwargs):
        assert npasses >= 1, "npasses should be >= 1, v/s {}".format(npasses)
        self.npasses = npasses

        assert shuffle_seed is None or isinstance(shuffle_seed, (int, np.int_)),\
                ("shuffle_seed should be either None (no shuffling)"
                 " or an integer, v/s {}".format(shuffle_seed))
        self.shuffle_seed = shuffle_seed

        self._pseeds = None  # sets the order of chunk reading
        self._corder = None

        self._cseeds = None  # sets the shuffling of arrays within a chunk

        self._input_shapes = None

        super(BaseInputsProvider, self).__init__(filepath, **kwargs)

    def _set_input_shapes(self):
        inputs = next(self.flow(indefinitely=False))

        self._input_shapes = [i.shape for i in inputs]

    @property
    def inputdatashape(self):
        if self._input_shapes is None:
            self._set_input_shapes()

        return (None, ) + self._input_shapes[0][1:]

    @property
    def inputlabelshape(self):
        if self._input_shapes is None:
            self._set_input_shapes()

        return (None, ) + self._input_shapes[1][1:]

    @property
    def steps_per_pass(self):
        return self.nchunks

    def _setup_shuffling_seeds(self):
        if self.shuffle_seed is None:
            self._pseeds = (None, ) * self.npasses
            self._corder = (np.arange(self.nchunks), ) * self.npasses
            self._cseeds = ((None, ) * self.nchunks, ) * self.npasses
        else:
            nseeds = self.npasses * (1 + self.nchunks)
            nr.seed(self.shuffle_seed)  # pylint: disable=no-member
            seeds = nr.randint(41184535, size=nseeds)  # pylint: disable=no-member

            self._pseeds = tuple(seeds[:self.npasses])

            _corder = []
            for s in self._pseeds:
                nr.seed(s)  # pylint: disable=no-member
                _corder.append(nr.permutation(self.nchunks))  # pylint: disable=no-member

            self._corder = tuple(_corder)
            self._cseeds = nu.totuples(seeds[self.npasses:].reshape(
                (self.npasses, -1)))

    def _chunk_order_for_pass(self, p):
        if self._corder is None:
            self._setup_shuffling_seeds()

        return self._corder[p]

    def _seed_for_chunk_in_pass(self, p, c):
        if self._cseeds is None:
            self._setup_shuffling_seeds()

        return self._cseeds[p][c]

    @staticmethod
    def maybe_shuffle_array(arr, shuffle_seed):
        if shuffle_seed is None:
            return arr
        elif isinstance(shuffle_seed, (int, np.int_)):
            nr.seed(shuffle_seed)  # pylint: disable=no-member
            nr.shuffle(arr)  # pylint: disable=no-member
            return arr
        else:
            raise ValueError(
                "shuffle_seed should either be None (no shuffling) or an integer"
                " {} was given {}".format(shuffle_seed, type(shuffle_seed)))

    def get_prepped_inputs(self,
                           chunking,
                           array_shuffle_seed=None,
                           only_labels=False,
                           **kwargs):
        """ The method that will be called by InputsProvider (below).

        Override if, for example, you want to also have sample_weights returned.
        However, other classes expect the first two arrays in the tuple to be
        data and label, in that order.
        """
        data, label = self.get_prepped_data_label(
            chunking, only_labels=only_labels, **kwargs)

        # NOTE: since the same shufflng seed is used for all arrays,
        # The correspondence will be preserved
        return (self.maybe_shuffle_array(data, array_shuffle_seed),
                self.maybe_shuffle_array(label, array_shuffle_seed))

    def flow_for_pass(self,
                      at,
                      starting_chunk_at=0,
                      only_labels=False,
                      with_chunking=False,
                      **kwargs):
        co = self._chunk_order_for_pass(at)
        ii = 0
        try:
            for i in range(starting_chunk_at, len(co)):
                ii = i
                c = co[i]
                chunking = self.chunkings[c]
                seed = self._seed_for_chunk_in_pass(at, c)

                inputs = self.get_prepped_inputs(
                    chunking=chunking,
                    array_shuffle_seed=seed,
                    only_labels=only_labels,
                    **kwargs)

                if with_chunking:
                    yield inputs, ((c, ), chunking)
                else:
                    yield inputs

        except GeneratorExit:
            return

        except:  # print info helpful to resume if any error happened
            s = ".".join((self.__module__.split('.')[-1],
                          self.__class__.__name__))
            print("{}: An Error has stopped the flow at:".format(s))
            print("pass: {}\nchunk: {}".format(at, ii))
            print("npasses: {}\nshuffle_seed: {}".format(
                self.npasses, self.shuffle_seed))
            print("soucefile:\n{}".format(self.filepath))
            raise

    def flow(  # pylint: disable=too-many-arguments
            self,
            indefinitely=False,
            starting_pass_at=0,
            starting_chunk_at=0,
            only_labels=False,
            only_data=False,
            with_chunking=False,
            **kwargs):
        while True:
            for p in range(starting_pass_at, self.npasses):
                for inputs in self.flow_for_pass(
                        at=p,
                        starting_chunk_at=starting_chunk_at,
                        only_labels=only_labels,
                        with_chunking=with_chunking,
                        **kwargs):
                    if only_data:
                        inputs = inputs[0]

                    yield inputs

                starting_chunk_at = 0

            starting_pass_at = 0

            if not indefinitely:
                break


class BaseClassSubsamplingInputsProvider(BaseInputsProvider):  # pylint: disable=abstract-method
    """ Subsamples the inputs based on class.

    The class is determined by classkeyfn applied to each prepped label.

    NOTE: Sub-sampling can be uniform-random or hopped (based on shuffle_seed), but
    it is always done within a contiguous segment of the label (contiguous set of
    prepped labels with the same result for classkeyfn). This may result is less
    subsampling (more number of samples in final output) than provided ratios.

    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            filepath,
            nclasses=3,
            classkeyfn=np.argmax,  # for categorical labels
            class_subsample_to_ratios=1.,  # float, tuple or dict, default keeps all
            shuffle_seed=None,
            npasses=1,
            **kwargs):

        acceptables = (int, np.int_, float, np.float_,
                       Iterable)  #, np.ndarray, dict)
        assert isinstance(class_subsample_to_ratios,
                          acceptables), ("class_subsample_to_ratios shou;d")
        self._user_ratios = class_subsample_to_ratios
        self._ratios = None
        self.nclasses = nclasses
        self.classkeyfn = classkeyfn

        super(BaseClassSubsamplingInputsProvider, self).__init__(
            filepath, shuffle_seed=shuffle_seed, npasses=npasses, **kwargs)

    def _make_ratios_dict(self):
        self._ratios = dict.fromkeys(range(self.nclasses), 1.)

        if not isinstance(self._user_ratios, Iterable):
            assert 0 <= self._user_ratios <= 1, (
                "subsampling ratios should be >= 0 and <= 1, "
                "v/s, {}".format(self._user_ratios))
            r = float(self._user_ratios)
            for k in self._ratios.keys():
                self._ratios[k] = r
        else:
            assert len(self._user_ratios) <= self.nclasses, (
                "The input labels have fewer classes "
                "than ratios are provided for, "
                "{} v/s {} in {}".format(self.nclasses,
                                         len(self._user_ratios),
                                         self._user_ratios))
            for i in range(len(self._user_ratios)):
                assert 0 <= self._user_ratios[i] <= 1, (
                    "subsampling ratios should be >= 0 and <= 1, "
                    "v/s, {}".format(self._user_ratios))
                self._ratios[i] = self._user_ratios[i]

    @property
    def ratios(self):
        if self._ratios is None:
            self._make_ratios_dict()

        return self._ratios

    @staticmethod
    def _prep_keep(segs_keeps, keep_seed=None, **kwargs):  # pylint: disable=unused-argument
        if keep_seed is None:
            return np.sort(
                np.concatenate([seg[:keep] for seg, keep in segs_keeps]))
        else:
            nr.seed(keep_seed)  # pylint: disable=no-member
            seeds = nr.randint(41184535, size=len(segs_keeps))  # pylint: disable=no-member

            keeps = []
            for i, (s, k) in enumerate(segs_keeps):
                if len(s) == k:
                    # we're keeping all
                    keeps.append(s)
                else:
                    nr.seed(seeds[i])  # pylint: disable=no-member
                    keeps.append(nr.permutation(s)[:k])  # pylint: disable=no-member

            # NOTE: We only do random sampling, not shuffling
            return np.sort(np.concatenate(keeps))

    def keeping_decision(self, inputs, keep_seed=None, **kwargs):
        labels = inputs[1]  # this is usually the case, esp for Keras inputs
        if all(l == 1. for l in self.ratios.values()):
            # we're keeping all ...
            # shuffling will happen on these keeps later
            return np.arange(len(labels))

        # Ref:
        # http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance
        # FIXME: Assumes categorical labels ... 2D
        diff = np.concatenate([
            np.ones((1, ) + labels.shape[1:], dtype=labels.dtype),
            np.diff(labels, axis=0),
        ])
        starts = np.unique(np.where(diff)[0])
        ends = np.concatenate([starts[1:], [len(labels)]])

        seg_keep = []
        for s, e, l in zip(starts, ends, labels[starts]):
            ratio = self.ratios[self.classkeyfn(l)]
            idx = np.arange(s, e)

            if ratio == 0.:  # skip all
                continue
            elif ratio == 1.:  # keep all
                seg_keep.append((idx, len(idx)))
            else:
                lenkeep = int(len(idx) * ratio)
                if lenkeep == 0:
                    # keep at least 1, cuz we are subsamping, not skipping
                    # NOTE: this may be controversial
                    lenkeep = 1
                seg_keep.append((idx, lenkeep))

        return self._prep_keep(seg_keep, keep_seed=keep_seed, **kwargs)

    def get_prepped_inputs(self, chunking, array_shuffle_seed=None, **kwargs):
        sup = super(BaseClassSubsamplingInputsProvider, self)
        inputs = sup.get_prepped_data_label(chunking, **kwargs)

        # shuffle seeds for order of steps, and shuffling data
        if array_shuffle_seed is None:
            kseed, aseed = None, None
        elif isinstance(array_shuffle_seed, (int, np.int_)):
            nr.seed(array_shuffle_seed)  # pylint: disable=no-member
            kseed, aseed = nr.randint(41184535, size=2)  # pylint: disable=no-member

        keeps = self.keeping_decision(inputs, keep_seed=kseed, **kwargs)

        keeps = self.maybe_shuffle_array(keeps, aseed)

        if len(keeps) == 0:
            warning("Sub-sampling has resulted in zero-length selection, "
                    "ratios used: {} from given {}".format(
                        self.ratios, self._user_ratios))

        return [i[keeps, ...] for i in inputs]


class BaseSteppedInputsProvider(BaseInputsProvider):  # pylint: disable=abstract-method
    """ Provider that flows a chunk in multiple steps.

    NOTE: While the data and order of steps may get shuffled (depends on shuffle_seed),
    consecutive steps will still come from the same chunk.
    """

    def __init__(self,
                 filepath,
                 steps_per_chunk=1,
                 shuffle_seed=None,
                 npasses=1,
                 **kwargs):
        assert steps_per_chunk >= 1, (
            "steps_per_chunk should be >= 1, v/s {}".format(steps_per_chunk))
        self.steps_per_chunk = steps_per_chunk

        super(BaseSteppedInputsProvider, self).__init__(
            filepath, npasses=npasses, shuffle_seed=shuffle_seed, **kwargs)

    @property
    def steps_per_pass(self):
        return self.steps_per_chunk * super(BaseSteppedInputsProvider,
                                            self).steps_per_pass

    def se_for_chunksteps_maybeshuffled(self,
                                        len_input,
                                        shuffle_seed=None,
                                        **kwargs):  # pylint: disable=unused-argument
        # don't want to kill the training man
        if len_input < self.steps_per_chunk:
            warning("chunk size is smaller than steps_per_chunk: "
                    "{} v/s {}, will use the smaller value".format(
                        len_input, self.steps_per_chunk))
            spc = len_input
        else:
            spc = self.steps_per_chunk

        si = len_input // spc

        starts = tuple(range(0, si * spc, si))
        ends = starts[1:] + (len_input, )

        # NOTE: I know that we can get pre-shuffled data from sup.get_prepped_inputs,
        # but ... This is to test an API, because ... there will be other implementations
        # that cannot shuffle the data as is ... like the ones that use striding tricks

        # shuffle seeds for order of steps, and shuffling data
        if shuffle_seed is None:
            oseed, aseed = None, None
        elif isinstance(shuffle_seed, (int, np.int_)):
            nr.seed(shuffle_seed)  # pylint: disable=no-member
            oseed, aseed = nr.randint(41184535, size=2)  # pylint: disable=no-member

        starts = self.maybe_shuffle_array(np.array(starts), oseed)
        ends = self.maybe_shuffle_array(np.array(ends), oseed)

        return starts, ends, aseed

    def get_prepped_inputs(self,
                           chunking,
                           array_shuffle_seed=None,
                           only_labels=False,
                           **kwargs):
        sup = super(BaseSteppedInputsProvider, self)
        inputs = sup.get_prepped_data_label(
            chunking, only_labels=only_labels, **kwargs)

        # assuming all inputs have the same length in the first dim (or else Keras will not accept)
        len_input = len(inputs[0])
        starts, ends, aseed = self.se_for_chunksteps_maybeshuffled(
            len_input, shuffle_seed=array_shuffle_seed, **kwargs)

        keeps = self.maybe_shuffle_array(np.arange(len_input), aseed)
        for s, e in zip(starts, ends):
            yield [i[keeps[s:e], ...] for i in inputs]

    def flow(  # pylint: disable=too-many-arguments, too-many-locals
            self,
            indefinitely=False,
            starting_pass_at=0,
            starting_chunk_at=0,
            only_labels=False,
            only_data=False,
            with_chunking=False,
            *args,
            **kwargs):
        sup = super(BaseSteppedInputsProvider, self)
        gen = sup.flow(
            indefinitely=indefinitely,
            starting_pass_at=starting_pass_at,
            starting_chunk_at=starting_chunk_at,
            only_labels=only_labels,
            with_chunking=with_chunking,
            *args,
            **kwargs)

        for stepped_ip in gen:
            if with_chunking:
                stepped_ip, (c, chunking) = stepped_ip

            for i, ip in enumerate(stepped_ip):
                if only_data:
                    ip = ip[0]

                if with_chunking:
                    yield ip, (c + (i, ), chunking)
                else:
                    yield ip


class BaseClassSubsamplingSteppedInputsProvider(  # pylint: disable=abstract-method
        BaseClassSubsamplingInputsProvider, BaseSteppedInputsProvider):
    """
    NOTE: Sub-sampling is done before making steps out of the chunk, hence it is
    still likely that there can be empty chunks, and hence,
    consecutive empty steps for such chunks.
    Yes, we will still honor the steps_per_chunk, even though the inputs provided
    will be of zero length. Don't worry ... they will still have the original shape,
    if at all the original prepped data had the necessary shape
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            filepath,
            nclasses=3,
            classkeyfn=np.argmax,  # for categorical labels
            class_subsample_to_ratios=1.,  # float, tuple or dict, default keeps all
            steps_per_chunk=1,
            shuffle_seed=None,
            npasses=1,
            **kwargs):

        # Mainly here for documentation and auto-completion
        sup = super(BaseClassSubsamplingSteppedInputsProvider, self)
        sup.__init__(
            filepath,
            nclasses=nclasses,
            classkeyfn=classkeyfn,
            class_subsample_to_ratios=class_subsample_to_ratios,
            steps_per_chunk=steps_per_chunk,
            shuffle_seed=shuffle_seed,
            npasses=npasses,
            **kwargs)

    def get_prepped_inputs(self, chunking, array_shuffle_seed=None, **kwargs):
        sup = super(BaseClassSubsamplingSteppedInputsProvider, self)
        inputs = sup.get_prepped_data_label(chunking, **kwargs)

        # shuffle seeds for order of steps, and shuffling keeps
        if array_shuffle_seed is None:
            kseed, seed = None, None
        elif isinstance(array_shuffle_seed, (int, np.int_)):
            nr.seed(array_shuffle_seed)  # pylint: disable=no-member
            kseed, seed = nr.randint(41184535, size=2)  # pylint: disable=no-member

        keeps = self.keeping_decision(inputs, keep_seed=kseed, **kwargs)
        if len(keeps) == 0:
            warning("Sub-sampling has resulted in zero-length selection, "
                    "ratios used: {} from given {}".format(
                        self.ratios, self._user_ratios))
            for _ in range(self.steps_per_chunk):
                # zero-length, but we honor the steps per chunk
                yield [i[keeps, ...] for i in inputs]

        starts, ends, aseed = self.se_for_chunksteps_maybeshuffled(
            len(keeps), shuffle_seed=seed, **kwargs)

        keeps = self.maybe_shuffle_array(keeps, aseed)
        for s, e in zip(starts, ends):
            yield [i[keeps[s:e], ...] for i in inputs]


class BaseWithContextSteppedInputsProvider(  # pylint: disable=abstract-method
        BaseWithContextPrepper, BaseSteppedInputsProvider):
    """
    NOTE: Assumes that the data and labels have been read and prepped, and
    that the labels are in categorical/softmax form.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            filepath,
            data_context=0,
            label_subcontext=0,  # 0 for choosing only the center label as label
            label_from_subcontext_fn=dominant_label_for_subcontext,
            add_channel_at_end=True,
            steps_per_chunk=8,
            shuffle_seed=None,
            npasses=1,
            **kwargs):
        sup = super(BaseWithContextSteppedInputsProvider, self)
        sup.__init__(
            filepath,
            data_context=data_context,
            label_subcontext=label_subcontext,
            label_from_subcontext_fn=label_from_subcontext_fn,
            add_channel_at_end=add_channel_at_end,
            steps_per_chunk=steps_per_chunk,
            shuffle_seed=shuffle_seed,
            npasses=npasses,
            **kwargs)

    def get_prepped_inputs(self,
                           chunking,
                           array_shuffle_seed=None,
                           only_labels=False,
                           **kwargs):
        # NOTE: In order to shuffle, the data from strided view has to be copied.
        # Otherwise, it will lead to corruption of the data.
        # WithContextInputsProvider is not implemented ...
        # because the entire chunk would have had to be copied ... nullifying the
        # benefits of using strided_view ...
        #
        # SteppedInputsProvider leads to copying anyway (by shuffling keeps),
        # and here, we avoid unnecessary copying when we don't have to shuffle
        # and for that, we can't use keeps, even unshuffled one,
        # because that would lead to copying.
        # And, since we have to implement this method for that, Python is not
        # cooperating with simply calling SteppedInputsProvider when shuffling,
        # and recursing indefinitely. Hence, this copied implementation

        sup = super(BaseWithContextSteppedInputsProvider, self)
        inputs = sup.get_prepped_data_label(
            chunking, only_labels=only_labels, **kwargs)

        len_input = len(inputs[0])
        if array_shuffle_seed is not None:
            starts, ends, aseed = self.se_for_chunksteps_maybeshuffled(
                len_input, shuffle_seed=array_shuffle_seed, **kwargs)

            keeps = self.maybe_shuffle_array(np.arange(len_input), aseed)
            for s, e in zip(starts, ends):
                yield [i[keeps[s:e], ...] for i in inputs]
        else:
            # no shuffling ... don't copy ...
            # used in validation inputs providers
            starts, ends, _ = self.se_for_chunksteps_maybeshuffled(
                len_input, shuffle_seed=None, **kwargs)

            for s, e in zip(starts, ends):
                yield [i[s:e, ...] for i in inputs]


class BaseWithContextClassSubsamplingSteppedInputsProvider(  # pylint: disable=abstract-method, too-many-ancestors
    BaseWithContextPrepper, BaseClassSubsamplingSteppedInputsProvider):
    def __init__(  # pylint: disable=too-many-arguments
            self,
            filepath,
            data_context=0,
            label_subcontext=0,  # 0 for choosing only the center label as label
            label_from_subcontext_fn=dominant_label_for_subcontext,
            add_channel_at_end=True,
            steps_per_chunk=8,
            classkeyfn=np.argmax,  # for categorical labels
            class_subsample_to_ratios=1.,  # float, tuple or dict, default keeps all
            shuffle_seed=None,
            npasses=1,
            **kwargs):

        # Mainly here for documentation and auto-completion
        sup = super(BaseWithContextClassSubsamplingSteppedInputsProvider, self)
        sup.__init__(
            filepath,
            data_context=data_context,
            label_subcontext=label_subcontext,
            label_from_subcontext_fn=label_from_subcontext_fn,
            add_channel_at_end=add_channel_at_end,
            classkeyfn=classkeyfn,
            class_subsample_to_ratios=class_subsample_to_ratios,
            steps_per_chunk=steps_per_chunk,
            shuffle_seed=shuffle_seed,
            npasses=npasses,
            **kwargs)

    def get_prepped_inputs(self,
                           chunking,
                           array_shuffle_seed=None,
                           only_labels=False,
                           **kwargs):
        # NOTE: In order to shuffle, the data from strided view has to be copied.
        # Otherwise, it will lead to corruption of the data.
        # WithContextInputsProvider is not implemented ...
        # because the entire chunk would have had to be copied ... nullifying the
        # benefits of using strided_view ...
        #
        # ClassSubsamplingSteppedInputsProvider leads to copying anyway (by shuffling keeps),
        # and here, we avoid unnecessary copying when we don't have to shuffle or subsample
        # and for that, we can't use keeps, even unshuffled one,
        # because that would lead to copying.
        # And, since we have to implement this method for that, Python is not
        # cooperating with simply calling SteppedInputsProvider when shuffling,
        # and recursing indefinitely. Hence, this copied implementation

        sup = super(BaseWithContextClassSubsamplingSteppedInputsProvider, self)
        inputs = sup.get_prepped_data_label(
            chunking, only_labels=only_labels, **kwargs)

        if array_shuffle_seed is not None or not all(
                v == 1. for v in self.ratios.values()):
            # there will be copying, whether due to shuffling or subsampling

            # shuffle seeds for order of steps, and shuffling keeps
            if array_shuffle_seed is None:
                kseed, seed = None, None
            elif isinstance(array_shuffle_seed, (int, np.int_)):
                nr.seed(array_shuffle_seed)  # pylint: disable=no-member
                kseed, seed = nr.randint(41184535, size=2)  # pylint: disable=no-member

            # decide which to keep
            keeps = self.keeping_decision(inputs, keep_seed=kseed, **kwargs)
            if len(keeps) == 0:
                warning("Sub-sampling has resulted in zero-length selection, "
                        "ratios used: {} from given {}".format(
                            self.ratios, self._user_ratios))
                for _ in range(self.steps_per_chunk):
                    # zero-length, but we honor the steps per chunk
                    yield [i[keeps, ...] for i in inputs]

            # decide stepping through the keeps
            starts, ends, seed = self.se_for_chunksteps_maybeshuffled(
                len(keeps), shuffle_seed=seed, **kwargs)

            # step through the keeps
            keeps = self.maybe_shuffle_array(keeps, seed)
            for s, e in zip(starts, ends):
                yield [i[keeps[s:e], ...] for i in inputs]

        else:
            # no shuffling ... no subsampling ... don't copy ...
            # used in validation inputs providers
            starts, ends, _ = self.se_for_chunksteps_maybeshuffled(
                len(inputs[0]), shuffle_seed=None, **kwargs)

            for s, e in zip(starts, ends):
                yield [i[s:e, ...] for i in inputs]


BaseWCtxSubsplStpdInputsProvider = BaseWithContextClassSubsamplingSteppedInputsProvider
BaseWCtxStpdInputsProvider = BaseWithContextSteppedInputsProvider
