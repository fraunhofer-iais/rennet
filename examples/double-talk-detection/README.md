# Double Talk Detection using [`RENNET`](https://github.com/fraunhofer-iais/rennet)

This repository contains the notebooks and scripts with the actual experiments and workflow steps carried out during Abdullah's research on Double-Talk Detection using Deep Learning during 2016-2017 period.
The outcomes of the research are part of a [full master's thesis](http://publica.fraunhofer.de/documents/N-477004.html).

The entire workflow has been implemented in Python.
`Keras` with `Tensorflow` as the backend was used to implement, train and evaluate the deep learning models.
`Numpy`, `LibRosa` and `Dask` are used for all the tasks surrounding that.

## Getting Started
All instructions throughout this README file assume you are using a UNIX like system and Bash.

1. Create a new Python virtual environment, preferably with Python version >= 3.5.0:
```
python3 -m venv ~/.virtualenvs/rennet
```
2. Activate the virtual environment:
```
source ~/.virtualenvs/rennet/bin/activate  
# or .../rennet/Scripts/activate on Windows
```
3. [Install `RENNET`](https://github.com/fraunhofer-iais/rennet/#installation) with at least the `[analysis]` extras.
4. Clone the [`RENNET`](https://github.com/fraunhofer-iais/rennet) repository locally if you haven't already.
5. Continue reading this README.

A successful installation will also have installed the necessary libraries, and would have instructed you in case of failure.

Please note that if a compatible version of Tensorflow was not already installed, a compatible version will be installed during setting up of `RENNET` and **will not** be a GPU-enabled.
You could either install a compatible version of Tensorflow that has GPU support _before_ installing `RENNET`,
or uninstall the Tensorflow version installed with `RENNET` setup, and install a GPU enabled one that is compatible. The best way to check would be to activate the virtual environment and

## Contents of This Folder
In any of the directories in this folder (except those that are empty directories):
- a subdirectory `dtfinale` will consist of the _actual_ scripts, notebooks, etc. (based on parent directory) that were used for Abdullah's research, and are here mainly for reference.
    + There is a good likelihood that these are broken since they were developed for a very specific environment.
- a subdirectory `fisher` will consist of scripts, notebooks, etc. that you can use to reproduce Abdullah's research if you have access to the Fisher English Corpus dataset (refer to the [thesis](http://publica.fraunhofer.de/documents/N-477004.html) for the exact details).
    + They are simplified and better consolidated versions of the similar scripts in their sister `dtfinale` directories.
    + The contents of these directories are what you will be using for various reproduction/inspiration steps later.

See [below](#annonet-py) for information on `annonet.py`.
It can be used to annotate arbitrary speech files using the models that will be trained over the course of the instructions below.
The annotation relies on model parameters other than the neural-network's weights that are defined and implemented in one of the `RennetModel`s in `models.py`.

### More About Some of the Main Directories
- `data`
    + Empty directory where all the datasets and data extraction will be organized later (if you choose to use it).
- `notebooks`
    + Home to all the executable [Jupyter (/IPython) notebooks](http://jupyter.org/).
    + They consist of all the main parts of the workflow.
    + These notebooks are reports of what was done, and you will have to _extract_ the instructions from it.
    + Information about the individual notebooks can be found [below](#notebooks-fisher).
- `outputs`
    + Empty directory where all the outputs from trainings will be organized by the training scripts (if you choose to use it).
- `scripts`
    + Home to all the scripts for training and evaluation.
    + `scripts/common.sh` sets up some common environment variables during trainings. You can also customize the variables if you know what you are doing.
    + You'll use `.sh` scripts in `fisher` to run the trainings of the models __AFTER__ the data has been prepared and features extracted following the notebooks in `notebooks/fisher`.
    + More discussion about them is done [below](#training).


#### `notebooks/fisher`
Jupyter would already have been installed as part of the `[analysis]` extras during [`RENNET` setup](#getting-started) above.

To work with the Jupyter notebooks:
- Open a new Bash session at **this** directory.
- Activate the Python virtual environment where `RENNET` was installed
- start jupyter notebook server:
```
jupyter notebook
```
- If not already launched, open the `localhost:XXXX` link shouted out by the command in a web-browser (e.g. Firefox).
- Navigate to `notebooks/fisher`, and open a notebook, preferably starting with first one and going in order as follows:
    1. **`01-data-acquisition-analysis-export_wav8kmono.ipynb`** consists of steps for setting up the Fisher Corpus data (assuming you have access to them), and exporting them in an appropriate audio format to a directory structure expected by later steps. The name of the file indicates what data set was analyzed, and what audio format it was exported to.
    2. **`02-feature-extraction_logmel64_win32ms_hop10ms-calc-viterbi-priors.ipynb`** consists of steps for extracting the acoustic features and the corresponding labels, and saving them to a _specifically_ structured HDF5 file appropriate for training using the scripts in `scripts/fisher`. The name of the file indicate what parameters were used in performing the steps.
    3. **`03-training-keras-models.ipynb`** consists of _information_ about what the training scripts in `scripts/fisher` do. How to run these scripts is discussed [below](#training).
    4. **`04-preparing-rennet-model.ipynb`** consists of steps for constructing a `rennet_model` from the trained Keras models.
        * This `rennet_model` (exported as `model.h5`) is used by `./annonet.sh`.
        * There also possible steps for tuning the parameters of the particular `rennet_model` that is constructing.
- Read, modify and/or execute the cells one by one.

## Training
The instructions below assume that:
- you have read everything above and followed the instructions where applicable.
- you have completed the steps till feature extraction (i.e. 1 and 2) in the [`notebooks/fisher`](#notebooks-fisher).
    + Or, at least have the necessary data files organized in the manner that is expected.
    + At least read the notebook`03-training-keras-models.ipynb` to get information about what the training scripts do.

### Running the Training Scripts
- Open **this** directory in the terminal.
- Create an alias to `~/.virtualenvs/rennet` into `./.virtualenv`:
```
ln -s ~/.virtualenvs/rennet ./.virtualenv
```
- Execute the desired training script, for example (THIS WILL KEEP THE TERMINAL BUSY):
```
./scripts/fisher/no-n/keepzero.sh
```
- Find the outputs of the training in the respective folder in `./outputs`, for example `./outputs/fisher/no-n`.
    + You will find more information in **`./notebooks/fisher/04-preparing-rennet-model.ipynb`**.

## `annonet.py`
The instructions below assume that:
- you have access to a trained and prepared `model.h5` after following _**all**_ the notebooks and instructions above.
- you have placed the `model.h5` at `./data/models/model.h5` with respect to the directory with this README.

**Please note** that the instructions below, then, only apply to the model you trained with the exact instructions above (exact with respect to various parameter values for audio export, parameter extraction, model configuration, etc.). You may, and probably should, create a new script with the appropriate values for different parameters that suit your customizations and needs if you made them.

Once you have the appropriate `model.h5` created and placed at an appropriate location,
you can use this trained model to annotate a given speech recording to detect and segment situations with
silence, single=speaker speech, and multi-speaker speech (AKA overlap or double-talk).
1. Open a new Bash session at **this** directory.
2. Activate the Python virtual environment where `RENNET` was installed.
3. Run the following to analyze two files for example:
```
python annonet.py /path/to/file1.wav /path/to/file2.mp4
```

### More help:
```
usage: python annonet.py [-h] [--todir [TODIR]] [--modelpath [MODELPATH]]
                         [--debug]
                         infilepaths [infilepaths ...]

Annotate some audio files using models trained with rennet.

positional arguments:
  infilepaths           Paths to input audio files to be analyzed

optional arguments:
  -h, --help            show this help message and exit
  --todir [TODIR]       Path to output directory. Will be created if it
                        doesn't exist (default: respective directories of the
                        inputfiles)
  --modelpath [MODELPATH], -M [MODELPATH]
                        Path to the model file (default:
                        ./data/models/model.h5). Please add if missing.
  --debug               Enable debugging mode where no errors are suppressed
                        during analysis.
```
