echo "Starting with r3"
conda create -n r3 python=3.5 numpy scipy
source activate r3
conda install -n r3 -c conda-forge numpy scipy tensorflow
conda install -n r3 -c conda-forge pip
conda install -n r3 -c conda-forge pytest
conda install -n r3 jupyter
conda install -n r3 -c conda-forge matplotlib
conda install -n r3 -c conda-forge pylint
pip install yapf
pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
conda install click
pip install pydub

echo "Done with r3"
echo "Starting with r2"
conda create -n r2 python=2.7 numpy scipy
source activate r2
conda install -n r2 -c conda-forge numpy scipy tensorflow pip pytest matplotlib pylint
conda install -n r2 jupyter
pip install yapf
pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
conda install -n r2 click
pip install pydub
