#!usr/bin/env bash
#PBS -l walltime=01:00:00
#PBS -l nodes=1:gpus=1
#PBS -l mem=16GB
#PBS -j oe

# !!! NOTE: Add the above lines in the final running script at the top with any updates to pass to qsub.
# And then source this script to setup some sensible defaults and environment variables.

# Setup CUDA ... stuff
CUDA_PATH=/usr/local/cuda-8.0
CUDNN_PATH=/opt/software/cudnn/cuda_v5.1

export CUDA_PATH
export CUDNN_PATH
export LD_LIBRARY_PATH=${CUDA_PATH}/extras/CUPTI/lib64:${CUDA_PATH}/lib64:${CUDNN_PATH}/lib64:$LD_LIBRARY_PATH

# check if RENNET_X_ROOT is setup by the callee environment already, else do it
if [[ -z "${RENNET_X_ROOT}" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export RENNET_X_ROOT="$DIR/.."
fi

export RENNET_DATA_ROOT="$RENNET_X_ROOT/data"
export RENNET_OUT_ROOT="$RENNET_X_ROOT/outputs"
export VIRTUALENV_ROOT="$RENNET_X_ROOT/.virtualenv"  # the one for training on the GPU

# get path to activate script for appropriate OS
# NOTE: still the bash version of the script!!
case $(uname) in
    CYGWIN*)    ACTIVATE="$VIRTUALENV_ROOT/Scripts/activate";;
    MINGW*)     ACTIVATE="$VIRTUALENV_ROOT/Scripts/activate";;
    Linux*)     ACTIVATE="$VIRTUALENV_ROOT/bin/activate";;
    Darwin*)    ACTIVATE="$VIRTUALENV_ROOT/bin/activate";;
    *)          ACTIVATE="__UNKNOWN"
esac

if [[ "$ACTIVATE" == "__UNKNOWN" ]]; then
    echo "ERROR: Could not determine path to activate script for unmae $(uname)."
    echo "       Please update the common.sh file to point to the right one."
    exit 1
fi

export ACTIVATE

export KERAS_BACKEND=tensorflow
