from __future__ import division, print_function
import os
import sys
import numpy as np
from glob import glob
from datetime import datetime as d
from keras.models import load_model

sys.path.append(os.environ['RENNET_ROOT'])
import rennet.utils.keras_utils as ku
import rennet.datasets.fisher as fe
import rennet.utils.h5_utils as hu

# CONFIGS ########################################################### CONFIGS #
ip = fe.UnnormedFrameWithContextInputsProvider
dctx = 10
add_channel_dim = True
lctx = 5  # center label
lctx_fn = hu.max_label_for_subcontext

steps_per_chunk = 8

trn_passes = 5
epochs_per_pass = steps_per_chunk

trn_callids = 'all'
val_callids = fe.chosen_val_callids
tst_callids = 'all'

trn_class_subsampling = 1.  # (0, )  # (0, 0.2)
val_class_subsampling = 1.  # (0, )
tst_class_subsampling = val_class_subsampling

initial_epoch = 0

trn_shuffle_seed = 32
verbose = 2
pickle_safe = True
max_q_size = 3 * steps_per_chunk + 1

# OUPUT DIR ###################################################### OUTPUT DIR #
activity_dir = os.environ['ACTIVITY_OUT_DIR']

starting_model_fp = None

# DATA SOURCE ################################################### DATA SOURCE #
data_root = os.environ['RENNET_DATA_ROOT']

pickles_root = glob(
    os.path.join(data_root, 'working', 'fisher', 'fe_03_p1', 'wav-8k-mono',
                 'pickles', '*logmel64*'))[0]

trn_h5 = os.path.join(pickles_root, 'trn.h5')
val_h5 = os.path.join(pickles_root, 'val.h5')
tst_h5 = os.path.join(pickles_root, 'val.h5')


# MAIN ################################################################# MAIN #
def get_model(input_shape, nclasses=3):
    if starting_model_fp is None:
        return ku.model_c3(input_shape, nclasses, compile_model=True)
    else:
        return load_model(starting_model_fp)


def main():
    print("\n", "/" * 120, "\n")
    print(d.now())
    print("\nOUTPUTS DIRECTORY:\n{}\n".format(activity_dir))

    print("\nTRN H5:\n{}\n".format(trn_h5))
    trn_ip = ip.for_callids(
        trn_h5,
        callids=trn_callids,
        data_context=dctx,
        add_channel_at_end=add_channel_dim,
        label_subcontext=lctx,
        label_from_subcontext_fn=lctx_fn,
        steps_per_chunk=steps_per_chunk,
        classkeyfn=np.argmax,  # for categorical labels
        class_subsample_to_ratios=trn_class_subsampling,
        shuffle_seed=trn_shuffle_seed,
        npasses=trn_passes, )

    print(
        "{}: max-totlen: {:,}; nchunks: {:,}; steps_per_pass: {:,}; npasses: {:,}".
        format("TRN", trn_ip.totlen, trn_ip.nchunks, trn_ip.steps_per_pass,
               trn_ip.npasses))
    print("data shape: {}; label shape: {}".format(trn_ip.inputdatashape,
                                                   trn_ip.inputlabelshape))

    print("\nVAL H5:\n{}\n".format(val_h5))
    val_ip = ip.for_callids(
        val_h5,
        callids=val_callids,
        data_context=dctx,
        add_channel_at_end=add_channel_dim,
        label_subcontext=lctx,
        label_from_subcontext_fn=lctx_fn,
        steps_per_chunk=steps_per_chunk,
        classkeyfn=np.argmax,  # for categorical labels
        class_subsample_to_ratios=val_class_subsampling,
        shuffle_seed=None,  # never shuffled
        npasses=1, )

    print(
        "{}: max-totlen: {:,}; nchunks: {:,}; steps_per_pass: {:,}; npasses: {:,}".
        format("VAL", val_ip.totlen, val_ip.nchunks, val_ip.steps_per_pass,
               val_ip.npasses))
    print("data shape: {}; label shape: {}".format(val_ip.inputdatashape,
                                                   val_ip.inputlabelshape))

    print("\nTST H5:\n{}\n".format(tst_h5))
    tst_ip = ip.for_callids(
        tst_h5,
        callids=tst_callids,
        data_context=dctx,
        add_channel_at_end=add_channel_dim,
        label_subcontext=lctx,
        label_from_subcontext_fn=lctx_fn,
        steps_per_chunk=steps_per_chunk,
        classkeyfn=np.argmax,  # for categorical labels
        class_subsample_to_ratios=tst_class_subsampling,
        shuffle_seed=None,  # never shuffled
        npasses=1, )

    print(
        "{}: max-totlen: {:,}; nchunks: {:,}; steps_per_pass: {:,}; npasses: {:,}".
        format("TST", tst_ip.totlen, tst_ip.nchunks, tst_ip.steps_per_pass,
               tst_ip.npasses))
    print("data shape: {}; label shape: {}".format(tst_ip.inputdatashape,
                                                   tst_ip.inputlabelshape))

    trn_gen = trn_ip.flow(
        indefinitely=True,
        only_labels=False,
        with_chunking=False, )
    nepochs = epochs_per_pass * trn_passes
    steps_per_epoch = (trn_passes * trn_ip.steps_per_pass) // nepochs

    val_gen = val_ip.flow(
        indefinitely=True,
        only_labels=False,
        with_chunking=False, )
    validation_steps = val_ip.steps_per_pass

    callbacks = ku.create_callbacks(
        val_ip,
        activity_dir,
        epochs_per_pass,
        verbose=verbose == 1,
        pickle_safe=pickle_safe,
        max_q_size=max_q_size)

    input_shape = trn_ip.inputdatashape
    model = get_model(input_shape)

    print("\n", "/" * 120, "\n")
    print("MODEL SUMMARY")
    model.summary()

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
        initial_epoch=initial_epoch, )

    print("\nTRAINING ENDED")
    print(d.now())

    print("\n", "/" * 120, "\n")
    print("PREDICTING ON TEST")
    ku.predict_on_inputs_provider(
        model,
        tst_ip,
        activity_dir, )

    print("\nDONE")
    print(d.now())

if __name__ == '__main__':
    main()
