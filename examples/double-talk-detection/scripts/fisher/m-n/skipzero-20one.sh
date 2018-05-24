#!/bin/bash
#PBS -l walltime=200:00:00
#PBS -l nodes=1:gpus=1
#PBS -l mem=16GB
#PBS -j oe
export ACTIVITY_NAME="m-n/skipzero-20one"  # mean-normalization, skip-silence, subsample-singlespeech

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export RENNET_X_ROOT="$DIR/../../.."
source "$RENNET_X_ROOT/scripts/common.sh"

# !!!!!!!!!!!!!!!!!!!! ANY CUSTOMIZATION TO COMMON.SH !!!!!!!!!!!!!!!!!!!!!!!!!
# export RENNET_DATA_ROOT="$RENNET_X_ROOT/data"
# export RENNET_OUT_ROOT="$RENNET_X_ROOT/outputs"
# export VIRTUALENV_ROOT="$RENNET_X_ROOT/.virtualenv"  # the one for training on the GPU
#
# export ACTIVATE="$VIRTUALENV_ROOT/bin/activate"
#
# export KERAS_BACKEND=tensorflow

# where the appropriate val.h5, trn.h5 and tst.h5 are.
# !!!!!!!!!!!!! CHANGE HERE IF THE LOCATION IS DIFFERENT !!!!!!!!!!!!!!!!!!!!!!!
export PICKLES_ROOT="$RENNET_DATA_ROOT/working/fisher/fe_03_p1/wav-8k-mono/pickles"
export PICKLES_DIR="$PICKLES_ROOT/$(ls -t "$PICKLES_ROOT" | head -1)"  # choosing latest


export ACTIVITY_OUT_DIR="$RENNET_OUT_ROOT/fisher/$ACTIVITY_NAME"
mkdir -p "$ACTIVITY_OUT_DIR"

source "$ACTIVATE"

TRAIN_EVAL_SCRIPT="$RENNET_X_ROOT/scripts/fisher/train_eval.py"

python "$TRAIN_EVAL_SCRIPT" &> "$ACTIVITY_OUT_DIR/logs.txt"
