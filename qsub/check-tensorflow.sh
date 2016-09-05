#!/bin/bash
QSUB_SCRIPTS_ROOT=./qsub
# shellcheck disable=SC1090,SC1091
source $QSUB_SCRIPTS_ROOT/common.sh
python $QSUB_SCRIPTS_ROOT/check-tensorflow.py > $QSUB_SCRIPTS_ROOT/outputs/check-tensorflow.sh.out
echo "$LD_LIBRARY_PATH" >> $QSUB_SCRIPTS_ROOT/outputs/check-tensorflow.sh.out

