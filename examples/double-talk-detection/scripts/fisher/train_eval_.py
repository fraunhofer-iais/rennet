"""
@motjuste

The single file that performs training and evaluation for different configurations.
The configuration to use is determined by the environment variable 'ACTIVITY_NAME'.
"""
from __future__ import division, print_function
import os
import sys
import numpy as np
from datetime import datetime as d
import tensorflow
from h5py import File as hFile

import rennet.utils.keras_utils as ku
import rennet.datasets.fisher as fe
import rennet.utils.h5_utils as hu
import rennet.utils.np_utils as nu
import rennet.utils.label_utils as lu

load_model = tensorflow.keras.models.load_model

# CONFIGS ########################################################### CONFIGS #
activity_name = os.environ['ACTIVITY_NAME']
norm, sub = activity_name.split("/")
# norm \in {"no-n", "m-n", "mv-n"}
# sub \in {"keepzero", "skipzero", "skipzero-20one"}

ip = fe.ChMVNFrameWithContextInputsProvider

data_context = 10
add_channel_dim = True
label_subcontext = 0  # center label
label_subcontext_fn = hu.dominant_label_for_subcontext
mean_it = norm in ("m-n", "mv-n")
std_it = norm == "mv-n"

steps_per_chunk = 8

trn_passes = 20 if sub != "skipzero-20one" else 40  # passes over the entire training data
epochs_per_pass = steps_per_chunk

trn_callids = 'all'
val_callids = fe.chosen_val_callids
tst_callids = 'all'

trn_class_subsampling = { #1.  # (0, )  # (0, 0.2)
    "keepzero": 1.,
    "skipzero": (0, ),
    "skipzero-20one": (0, 0.2),
}[sub]
val_class_subsampling = {  # only skip (to reduce training time), but never sub-sample
    "keepzero": 1.,
    "skipzero": (0, ),
    "skipzero-20one": (0, ),
}[sub]
tst_class_subsampling = 1.  # never skip or sub-sample test

initial_epoch = 0

trn_shuffle_seed = 32
verbose = 2
pickle_safe = True
max_q_size = 3 * steps_per_chunk + 1

# OUPUT DIR ###################################################### OUTPUT DIR #
activity_dir = os.environ['ACTIVITY_OUT_DIR']

# DATA SOURCE ################################################### DATA SOURCE #
pickles_root = os.environ['PICKLES_DIR']

trn_h5 = os.path.join(pickles_root, 'trn.h5')
val_h5 = os.path.join(pickles_root, 'val.h5')
tst_h5 = os.path.join(pickles_root, 'tst.h5')

# MODEL ############################################################### MODEL #
starting_model_fp = None  # or path to an existing starting model.h5

# For eval only run:
# set path to existing model above, and set trn_passes to zero

# trn_passes = 0


def get_model(input_shape, nclasses=3):
    if starting_model_fp is None and trn_passes != 0:  # not an eval only script
        return ku.model_c3(input_shape, nclasses, compile_model=True)
    else:
        return load_model(starting_model_fp)


# HELPERS ############################################################ HELPERS #
def read_normalized_viterbi_priors():
    # read raw priors from val.h5 and trn.h5
    with hFile(val_h5, 'r') as f:
        vinit = f["viterbi/init"][()]
        vtran = f["viterbi/tran"][()]

    with hFile(trn_h5, 'r') as f:
        tinit = f["viterbi/init"][()]
        ttran = f["viterbi/tran"][()]

    init = vinit + tinit
    tran = vtran + ttran

    return lu.normalize_raw_viterbi_priors(init, tran)


def predict_on_inputs_provider(  # pylint: disable=too-many-locals,too-many-statements
        model, inputs_provider, export_to, init, tran):
    def _save(paths, datas):
        with hFile(export_to, 'a') as f:
            for path, data in zip(paths, datas):
                if path not in f.keys():
                    f.create_dataset(path, data=data, compression='lzf', fletcher32=True)

            f.flush()

    currn = None
    ctrue = []
    cpred = []

    tot_conf = None
    tot_conf_vp = None
    for xy, (_, chunking) in inputs_provider.flow(
        indefinitely=False,
        only_labels=False,
        with_chunking=True,
    ):

        ctrue.append(xy[1])
        cpred.append(model.predict_on_batch(xy[0]))

        if currn is None:
            currn = chunking.labelpath
            continue

        if chunking.labelpath != currn:
            t = np.concatenate(ctrue[:-1])
            p = np.concatenate(cpred[:-1])

            if sub != 'keepzero':  # from activity_name above
                z = t[:, 0].astype(bool)
                p[z, 0] = 1.
                p[z, 1:] = 0.

            # raw confusion
            conf = nu.confusion_matrix_forcategorical(
                t, nu.to_categorical(p.argmax(axis=-1), nclasses=t.shape[-1])
            )

            # viterbi decoded - no scaling
            vp = lu.viterbi_smoothing(p, init, tran)
            conf_vp = nu.confusion_matrix_forcategorical(
                t, nu.to_categorical(vp, nclasses=t.shape[-1])
            )

            _save(
                paths=["{}/{}".format(_p, currn) for _p in ('raw', 'viterbi')],
                datas=[conf, conf_vp],
            )

            print(currn, end=' ')
            nu.print_prec_rec(*nu.normalize_confusion_matrix(conf), onlydiag=True)

            if tot_conf is None:
                tot_conf = conf
                tot_conf_vp = conf_vp
            else:
                tot_conf += conf
                tot_conf_vp += conf_vp

            currn = chunking.labelpath
            ctrue = ctrue[-1:]
            cpred = cpred[-1:]

    # last chunking
    t = np.concatenate(ctrue)
    p = np.concatenate(cpred)

    if sub != 'keepzero':  # from activity_name above
        z = t[:, 0].astype(bool)
        p[z, 0] = 1.
        p[z, 1:] = 0.

    conf = nu.confusion_matrix_forcategorical(
        t, nu.to_categorical(p.argmax(axis=-1), nclasses=t.shape[-1])
    )

    vp = lu.viterbi_smoothing(p, init, tran)
    conf_vp = nu.confusion_matrix_forcategorical(
        t, nu.to_categorical(vp, nclasses=t.shape[-1])
    )

    _save(
        paths=["{}/{}".format(_p, currn) for _p in ('raw', 'viterbi')],
        datas=[conf, conf_vp],
    )

    print(currn, end=' ')
    nu.print_prec_rec(*nu.normalize_confusion_matrix(conf), onlydiag=True)

    tot_conf += conf
    tot_conf_vp += conf_vp

    # print out total-statistics
    _save(
        paths=["{}/{}".format(_p, 'final') for _p in ('raw', 'viterbi')],
        datas=[tot_conf, tot_conf_vp],
    )

    print("\nFINAL - RAW", end=' ')
    nu.print_prec_rec(*nu.normalize_confusion_matrix(tot_conf), onlydiag=False)

    print("\nFINAL - VITERBI", end=' ')
    nu.print_prec_rec(*nu.normalize_confusion_matrix(tot_conf_vp), onlydiag=False)


# MAIN ################################################################## MAIN #
def main():
    print("\n", "/" * 120, "\n")
    print(d.now())
    print(
        "\n\nCHUNKWISE", {
            "no-n": "NON-NORMALIZED",
            "m-n": "MEAN-NORMALIZED",
            "mv-n": "MEAN-VARIANCE-NORMALIZED",
        }[norm], {
            "keepzero": "WITH-SILENCE",
            "skipzero": "WITHOUT-SILENCE",
            "skipzero-20one": "WITHOUT-SILENCE-SUBSAMPLED-SINGLE-SPEECH",
        }[sub], "\n\n"
    )
    print("\nOUTPUTS DIRECTORY:\n{}\n".format(activity_dir))

    # Create input providers and shout a bunch of things ######################
    print("\nTRN H5:\n{}\n".format(trn_h5))
    trn_ip = ip.for_callids(
        trn_h5,
        callids=trn_callids,
        data_context=data_context,
        add_channel_at_end=add_channel_dim,
        label_subcontext=label_subcontext,
        label_from_subcontext_fn=label_subcontext_fn,
        steps_per_chunk=steps_per_chunk,
        classkeyfn=np.argmax,  # for categorical labels
        class_subsample_to_ratios=trn_class_subsampling,
        shuffle_seed=trn_shuffle_seed,
        npasses=trn_passes,
        mean_it=mean_it,
        std_it=std_it,
    )

    print(
        "{}: max-totlen: {:,}; nchunks: {:,}; steps_per_pass: {:,}; npasses: {:,}".format(
            "TRN", trn_ip.totlen, trn_ip.nchunks, trn_ip.steps_per_pass, trn_ip.npasses
        )
    )
    print(
        "data shape: {}; label shape: {}".format(
            trn_ip.inputdatashape, trn_ip.inputlabelshape
        )
    )

    print("\nVAL H5:\n{}\n".format(val_h5))
    val_ip = ip.for_callids(
        val_h5,
        callids=val_callids,
        data_context=data_context,
        add_channel_at_end=add_channel_dim,
        label_subcontext=label_subcontext,
        label_from_subcontext_fn=label_subcontext_fn,
        steps_per_chunk=steps_per_chunk,
        classkeyfn=np.argmax,  # for categorical labels
        class_subsample_to_ratios=val_class_subsampling,
        shuffle_seed=None,  # never shuffled
        npasses=1,
        mean_it=mean_it,
        std_it=std_it,
    )

    print(
        "{}: max-totlen: {:,}; nchunks: {:,}; steps_per_pass: {:,}; npasses: {:,}".format(
            "VAL", val_ip.totlen, val_ip.nchunks, val_ip.steps_per_pass, val_ip.npasses
        )
    )
    print(
        "data shape: {}; label shape: {}".format(
            val_ip.inputdatashape, val_ip.inputlabelshape
        )
    )

    print("\nTST H5:\n{}\n".format(tst_h5))
    tst_ip = ip.for_callids(
        tst_h5,
        callids=tst_callids,
        data_context=data_context,
        add_channel_at_end=add_channel_dim,
        label_subcontext=label_subcontext,
        label_from_subcontext_fn=label_subcontext_fn,
        steps_per_chunk=steps_per_chunk,
        classkeyfn=np.argmax,  # for categorical labels
        class_subsample_to_ratios=tst_class_subsampling,
        shuffle_seed=None,  # never shuffled
        npasses=1,
        mean_it=mean_it,
        std_it=std_it,
    )

    print(
        "{}: max-totlen: {:,}; nchunks: {:,}; steps_per_pass: {:,}; npasses: {:,}".format(
            "TST", tst_ip.totlen, tst_ip.nchunks, tst_ip.steps_per_pass, tst_ip.npasses
        )
    )
    print(
        "data shape: {}; label shape: {}".format(
            tst_ip.inputdatashape, tst_ip.inputlabelshape
        )
    )

    init, tran = read_normalized_viterbi_priors()
    print("GOT VITERBI PRIORS")
    # Setup stuff for training with keras #####################################
    trn_gen = trn_ip.flow(
        indefinitely=True,
        only_labels=False,
        with_chunking=False,
    )
    nepochs = epochs_per_pass * trn_passes
    steps_per_epoch = (
        trn_passes * trn_ip.steps_per_pass
    ) // nepochs if trn_passes != 0 else 0

    val_gen = val_ip.flow(
        indefinitely=True,
        only_labels=False,
        with_chunking=False,
    )
    validation_steps = val_ip.steps_per_pass

    callbacks = ku.create_callbacks(
        val_ip,
        activity_dir,
        epochs_per_pass,
        verbose=verbose == 1,
        pickle_safe=pickle_safe,
        max_q_size=max_q_size
    )

    input_shape = trn_ip.inputdatashape
    model = get_model(input_shape)

    print("\n", "/" * 120, "\n")
    print("MODEL SUMMARY")
    model.summary()

    if steps_per_epoch != 0:
        print("\n", "/" * 120, "\n")
        print("TRAINING BEGINS\n")
        model.fit_generator(
            trn_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=nepochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose,
            pickle_safe=pickle_safe,
            max_q_size=max_q_size,
            initial_epoch=initial_epoch,
        )

        print("\nTRAINING ENDED")
        print(d.now())

    print("\n", "/" * 120, "\n")
    print("PREDICTING ON TEST")
    export_to = os.path.join(activity_dir, "confs.test.h5")
    predict_on_inputs_provider(model, tst_ip, export_to, init, tran)

    print("\nDONE")
    print(d.now())


if __name__ == '__main__':
    main()
