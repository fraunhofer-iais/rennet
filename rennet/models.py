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
"""Module with rennet models

@motjuste
Created: 09-11-2017
"""
from __future__ import print_function, division, absolute_import
import copy
import os
from itertools import chain, repeat
import numpy as np
from h5py import File as hFile
from keras.models import load_model

from .utils import model_utils as mu
from .utils import audio_utils as au
from .utils import np_utils as nu
from .utils import label_utils as lu
from .utils.py_utils import makedirs_with_existok

# IDEA: Instead of hard-coded classes, serialize everything in the `model.h5`,
# and use a generic `rennet_model` class that can deserialize and create the appropriate
# class instance.
# Inspiration: Keras.
# Problems: Time, Worthiness for such limited set of tasks, Debugging.


# DOUBLE TALK DETECTION #######################################################
class DT_2_nosub_0zero20one_mono_mn(mu.BaseRennetModel):  # pylint: disable=too-many-instance-attributes, invalid-name
    __version__ = '0.1.0'

    def __init__(self, model_fp):
        # loading audio
        self.samplerate = 8000
        self.mono = True
        self.loadaudio = lambda fp: au.load_audio(
            filepath=fp,
            samplerate=self.samplerate,
            mono=self.mono, )

        # feature extraction
        self.win_len = int(self.samplerate * 0.032)
        self.hop_len = int(self.samplerate * 0.010)
        self.window = 'hann'
        self.n_mels = 64
        self.exttractfeat = lambda y: au.logmelspectrogram(
            y=y,
            sr=self.samplerate,
            n_fft=self.win_len,
            hop_len=self.hop_len,
            n_mels=self.n_mels,
            window=self.window
        )

        # feature normalization
        self.std_it = False
        self.norm_winsec = 200
        self.norm_winlen = int(
            (self.hop_len / self.samplerate * 1000) * self.norm_winsec
        )  # seconds
        self.first_mean_var = 'copy'
        self.normalize = lambda feat: nu.normalize_mean_std_rolling(
            feat,
            win_len=self.norm_winlen,
            std_it=self.std_it,
            first_mean_var=self.first_mean_var
        )

        # adding data-context
        self.data_context = 21
        self.addcontext = lambda x: nu.strided_view(
            x,
            win_shape=self.data_context,
            step_shape=1
        )

        # input generator
        self.batchsize = 256
        self.get_inputsgenerator = lambda X: (
            len(X) // self.batchsize + int(len(X) % self.batchsize != 0),  # nsteps
            (
                [x[..., None], x[..., None]] for x in chain(
                    nu.strided_view(X, win_shape=self.batchsize, step_shape=self.batchsize),
                    repeat(X[-(len(X) - self.batchsize * (len(X) // self.batchsize)):, ...])
                )
            )
        )

        # predict
        self.model = load_model(model_fp)
        self.verbose = 0
        self.max_q_size = 4

        # merging preds
        self.mergepreds_weights = np.array([[2, 2, 3], [0, 1, 1]])
        self.mergepreds_fn = lambda preds: mu.mergepreds_avg(preds, weights=self.mergepreds_weights)

        # viterbi smoothing
        with hFile(model_fp, 'r') as f:
            self.rinit = f['rennet/model/viterbi/init'][()]
            self.rtran = f['rennet/model/viterbi/tran'][()]
        self.vinit, self.vtran = lu.normalize_raw_viterbi_priors(self.rinit, self.rtran)

        # output
        self.seq_minstart = (
            self.win_len // 2 +  # removed during feature extraction
            int((self.data_context // 2) * self.hop_len)  # rm during adding data-context
        ) / self.hop_len  # bringing to hop's samplerate. NOTE: yes, this can float
        self.seq_samplerate = self.samplerate // self.hop_len
        self.seq_keep = 'keys'
        self.label_tiers = {
            0: "pred_none",
            1: "pred_single",
            2: "pred_multiple",
        }
        self.seq_annotinfo_fn = lambda label: lu.EafAnnotationInfo(
            tier_name=self.label_tiers[label]
        )
        self._cached_preds = dict()

        # get and set any params defined in the model_fp
        with hFile(model_fp, 'r') as f:
            model_group = f['rennet/model']
            print()
            for att in model_group.keys():
                if att == 'viterbi':
                    continue
                elif att in self.__dict__:
                    val = model_group[att][()]
                    prev = getattr(self, att)
                    setattr(self, att, val)

                    # IDEA: move this to __setattr__ method to shout-out **all** changes.
                    #       It will shout even on __init__ then,
                    #       which will have to be handled appropriately.
                    print(
                        "{}.{} updated from model file, from {} to {}".format(
                            self.__class__.__name__, att, prev, val
                        )
                    )

                # IDEA: Should we be pesky and raise errors when
                # there are unavailable `att` in the model file?
            print()

    def preprocess(self, filepath, **kwargs):  # pylint: disable=arguments-differ
        data = self.loadaudio(filepath)
        data = self.exttractfeat(data)
        data = self.normalize(data)
        return self.addcontext(data)

    def predict(self, X, model_fp=None, **kwargs):  # pylint: disable=arguments-differ
        nsteps, x_gen = self.get_inputsgenerator(X)

        model = self.model if model_fp is None else load_model(model_fp)

        return model.predict_generator(
            x_gen, steps=nsteps, verbose=self.verbose, max_q_size=self.max_q_size
        )

    def postprocess(self, preds, **kwargs):  # pylint: disable=arguments-differ
        pred = self.mergepreds_fn(preds)
        return lu.viterbi_smoothing(pred, self.vinit, self.vtran)

    def output(self, pred, to_filepath, audio_path=None, **kwargs):  # pylint: disable=arguments-differ
        seq = lu.ContiguousSequenceLabels.from_dense_labels(
            pred,
            keep=self.seq_keep,
            min_start=self.seq_minstart,
            samplerate=self.seq_samplerate
        )

        seq.to_eaf(
            to_filepath=to_filepath,
            linked_media_filepath=audio_path,
            annotinfo_fn=self.seq_annotinfo_fn
        )

    def apply(  # pylint: disable=too-many-arguments
            self,
            filepath,
            to_dir=None,
            to_fileextn=".preds.eaf",
            use_cached_preds=None,
            return_pred=False,
            **kwargs):
        filepath = os.path.abspath(filepath)
        if to_dir is None:
            to_dir = os.path.dirname(filepath)

        makedirs_with_existok(to_dir, exist_ok=True)

        to_filename = os.path.basename(filepath) + to_fileextn
        to_filepath = os.path.join(to_dir, to_filename)

        if use_cached_preds and filepath in self._cached_preds:
            x = self._cached_preds
        else:
            x = self.preprocess(filepath, **kwargs)
            x = self.predict(x, **kwargs)
            if use_cached_preds:
                self._cached_preds[filepath] = copy.deepcopy(x)

        x = self.postprocess(x, **kwargs)
        self.output(x, to_filepath, audio_path=filepath)

        return (to_filepath, x) if return_pred else to_filepath

    def export(self, *args, **kwargs):  # pylint: disable=arguments-differ
        raise NotImplementedError

    @classmethod
    def create_export(cls, model1, model2, viterbi_raw_init, viterbi_raw_tran):
        raise NotImplementedError


def get(identifier):
    return globals()[identifier]
