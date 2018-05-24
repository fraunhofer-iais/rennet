from __future__ import print_function, division
import os
import sys
import numpy as np
from glob import glob
from datetime import datetime as d
from h5py import File as hFile
from keras.models import load_model

sys.path.append(os.environ['RENNET_ROOT'])
# import rennet.utils.keras_utils as ku
import rennet.datasets.fisher as fe
import rennet.utils.h5_utils as hu
import rennet.utils.np_utils as nu

activity_name = os.environ['ACTIVITY_NAME']
activity_dir = os.environ['ACTIVITY_OUT_DIR']

starting_model_fp = os.path.join(activity_dir, activity_name, 'model.h5')
kmodel = load_model(starting_model_fp)

priors_fp = os.path.join(activity_dir, 'priors.h5')

# DATA SOURCE ################################################### DATA SOURCE #
data_root = os.environ['RENNET_DATA_ROOT']

pickles_root = glob(
    os.path.join(data_root, 'working', 'ka3', 'deutsch-00', 'wav-8k-mono',
                 'pickles', '*logmel64*'))[0]

# trn_h5 = os.path.join(pickles_root, 'trn.h5')
# val_h5 = os.path.join(pickles_root, 'val.h5')
# tst_h5 = os.path.join(pickles_root, 'tst.h5')
tst_h5 = os.path.join(pickles_root, 'val.h5')

# CONFIGS ########################################################### CONFIGS #
norm, sub = activity_name.split("_")
ip = {
    "un": fe.UnnormedFrameWithContextInputsProvider,
    "mn": fe.ChMVNFrameWithContextInputsProvider,
    "mvn": fe.ChMVNFrameWithContextInputsProvider,
}[norm]
dctx = 10
add_channel_dim = True
lctx = 0  # center label
lctx_fn = hu.dominant_label_for_subcontext
mean_it = norm != "un"
std_it = norm == "mvn"

steps_per_chunk = 8

tst_callids = 'all'

tst_class_subsampling = 1.


# FUNCS ############################################################### FUNCS #
def viterbi(obs, init, tran, priors=None):
    trellis = np.zeros(obs.shape)
    backpt = np.ones(obs.shape, dtype=int) * -1
    if priors is not None:
        obs = obs / priors
        obs = obs / obs.sum(axis=1)[:, None]

    obs = np.log(obs + 1e-15)  # pylint: disable=no-member
    init = np.log(init + 1e-15)  # pylint: disable=no-member
    tran = np.log(tran + 1e-15)  # pylint: disable=no-member

    trellis[0, :] = init * obs[0, ...]
    trellis_last = init * obs[0, ...]
    for t in range(1, len(obs)):
        x = trellis_last[None, ...] + tran
        backpt[t, :] = np.argmax(x, axis=1)
        trellis_last = np.max(x, axis=1) + obs[t, ...]

    tokens = [trellis_last.argmax()]
    for i in range(len(obs) - 1, 0, -1):
        tokens.append(backpt[i, tokens[-1]])

    return np.array(tokens[::-1])


def predict_on_inputs_provider(model, inputs_provider, subsampling,
                               export_to_dir, init, tran, priors):
    export_to = os.path.join(export_to_dir, "confs.h5")

    def _save(paths, datas):
        with hFile(export_to, 'a') as f:
            for path, data in zip(paths, datas):
                if path not in f.keys():
                    f.create_dataset(
                        path, data=data, compression='lzf', fletcher32=True)

            f.flush()

    currn = None
    ctrue = []
    cpred = []

    tot_conf = None
    tot_conf_vp = None
    tot_conf_svp = None
    for xy, (_, chunking) in inputs_provider.flow(
            indefinitely=False,
            only_labels=False,
            with_chunking=True, ):

        ctrue.append(xy[1])
        cpred.append(model.predict_on_batch(xy[0]))

        if currn is None:
            currn = chunking.labelpath
            continue

        if chunking.labelpath != currn:
            t = np.concatenate(ctrue[:-1])
            p = np.concatenate(cpred[:-1])

            if subsampling != 'nosub':
                z = t[:, 0].astype(bool)
                p[z, 0] = 1.
                p[z, 1:] = 0.

            # raw confusion
            conf = nu.confusion_matrix_forcategorical(
                t, nu.to_categorical(p.argmax(axis=-1), nclasses=t.shape[-1]))

            # viterbi decoded - no scaling
            vp = viterbi(p, init, tran, priors=None)
            conf_vp = nu.confusion_matrix_forcategorical(
                t, nu.to_categorical(vp, nclasses=t.shape[-1]))

            # viterbi decoded - scaling
            vp = viterbi(p, init, tran, priors=priors)
            conf_svp = nu.confusion_matrix_forcategorical(
                t, nu.to_categorical(vp, nclasses=t.shape[-1]))

            _save(
                paths=[
                    "{}/{}".format(_p, currn)
                    for _p in ('raw', 'viterbi', 'sviterbi')
                ],
                datas=[conf, conf_vp, conf_svp], )

            print(currn, end=' ')
            nu.print_prec_rec(
                *nu.normalize_confusion_matrix(conf), onlydiag=True)

            if tot_conf is None:
                tot_conf = conf
                tot_conf_vp = conf_vp
                tot_conf_svp = conf_svp
            else:
                tot_conf += conf
                tot_conf_vp += conf_vp
                tot_conf_svp += conf_svp

            currn = chunking.labelpath
            ctrue = ctrue[-1:]
            cpred = cpred[-1:]

    _save(
        paths=[
            "{}/{}".format(_p, 'final')
            for _p in ('raw', 'viterbi', 'sviterbi')
        ],
        datas=[tot_conf, tot_conf_vp, tot_conf_svp], )

    print("\nFINAL - RAW", end=' ')
    nu.print_prec_rec(*nu.normalize_confusion_matrix(tot_conf), onlydiag=False)

    print("\nFINAL - VITERBI", end=' ')
    nu.print_prec_rec(
        *nu.normalize_confusion_matrix(tot_conf_vp), onlydiag=False)

    print("\nFINAL - VITERBI - SCALED", end=' ')
    nu.print_prec_rec(
        *nu.normalize_confusion_matrix(tot_conf_svp), onlydiag=False)


def main():
    print("\n", "/" * 120, "\n")
    print(d.now())
    print("\nOUTPUTS DIRECTORY:\n{}\n".format(activity_dir))

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
        npasses=1,
        mean_it=mean_it,
        std_it=std_it, )

    print(
        "{}: max-totlen: {:,}; nchunks: {:,}; steps_per_pass: {:,}; npasses: {:,}".
        format("TST", tst_ip.totlen, tst_ip.nchunks, tst_ip.steps_per_pass,
               tst_ip.npasses))
    print("data shape: {}; label shape: {}".format(tst_ip.inputdatashape,
                                                   tst_ip.inputlabelshape))

    with hFile(priors_fp, 'r') as pf:
        init = np.array(pf['init'])
        tran = np.array(pf['tran'])
        priors = np.array(pf['priors'])

    init = init / init.sum()
    tran = nu.normalize_confusion_matrix(tran)[1]
    priors = priors / priors.sum()

    print("\n", "/" * 120, "\n")
    print("DONE SETTING PRIORS - NOW, MODEL SUMMARY")
    kmodel.summary()

    print("\n", "/" * 120, "\n")
    print("PREDICTING ON TEST")
    predict_on_inputs_provider(
        model=kmodel,
        inputs_provider=tst_ip,
        subsampling=sub,
        export_to_dir=os.path.join(activity_dir, activity_name),
        init=init,
        tran=tran,
        priors=priors, )

    print("\nDONE")
    print(d.now())


if __name__ == '__main__':
    main()
