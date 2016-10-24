#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64/:/opt/software/cudnn/cudnn_v4.0/lib64/

export RENNET_ROOT=/home/aabdullah/delve/rennet

python $RENNET_ROOT/rennet/models/20161024-gertv1k-0.py

