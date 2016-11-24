#!/bin/bash

echo "Starting with r3"
conda create -n r3 python=3.5 numpy scipy
source activate r3
conda install -n r3 numpy scipy pip pytest matplotlib pylint pytest jupyter click scikit-learn
conda install -n r3 -c conda-forge tensorflow
pip install yapf
pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
pip install pydub
pip install librosa
pip install keras
pip install tqdm

echo "Done with r3"
echo "Starting with r2"
conda create -n r2 python=2.7 numpy scipy
source activate r2
conda install -n r2 numpy scipy pip pytest matplotlib pylint pytest jupyter click scikit-learn
conda install -n r2 -c conda-forge tensorflow
pip install yapf
pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
pip install pydub
pip install librosa
pip install keras
pip install tqdm
