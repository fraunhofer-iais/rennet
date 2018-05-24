#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l nodes=1:gpus=1
#PBS -l mem=16GB
#PBS -j oe
source "$PWD/scripts/common.sh"  # !!! PWD=$RENNET_X_ROOT that should be explicit

# ANY CUSTOMIZATION TO COMMON.SH
# export RENNET_ROOT=/nm-raid/audio/work/abdullah/nm-rennet/rennet
#
# export RENNET_X_ROOT=/nm-raid/audio/work/abdullah/nm-rennet/rennet-x
#
# export RENNET_DATA_ROOT="$RENNET_X_ROOT/data"
# export RENNET_OUT_ROOT="$RENNET_X_ROOT/outputs"
#
# export VIRTUALENV_ROOT="$RENNET_X_ROOT/.virtualenv"


export ACTIVITY_NAME="mn_skipzero21"
export ACTIVITY_OUT_DIR="$RENNET_OUT_ROOT/eval/"
mkdir -p "$ACTIVITY_OUT_DIR/$ACTIVITY_NAME"

source $VIRTUALENV_ROOT/bin/activate
export PYTHONPATH=$RENNET_ROOT:$PYTHONPATH

python "$PWD/scripts/eval/eval.py" &> "$ACTIVITY_OUT_DIR/$ACTIVITY_NAME/logs.txt"
