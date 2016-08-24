#!/bin/bash
QSUB_SCRIPTS_ROOT=./qsub
. $QSUB_SCRIPTS_ROOT/common.sh
python $QSUB_SCRIPTS_ROOT/check-tensorflow.py &> $QSUB_SCRIPTS_ROOT/outputs/check-tensorflow.sh.out
