# RENNET

> *ˈrɛnɪt*
>
> curdled milk from the stomach of an unweaned calf, containing rennin and used in curdling milk for cheese.

## Usage Instructions for Analyzing Speech Recordings

1. In the terminal, change to the root of the local copy of this repository.
2. Run the following to analyze the two files, for example.
```
./annonet.sh path/to/file1.wav path/to/file2.mp4
```

> The first time the command is run, a local environment will be setup, which requires `git`, a connection to the internet and some time. Subsequent runs will not require this setup.

### Creating an Alias
1. Add the following to your `~/.bash_profile`, with the appropriate `path/to/local-rennet-repo`:
```
export RENNET_ROOT="path/to/local-rennet-repo"  # Change Here !!!
alias annonet="./$RENNET_ROOT/annonet.sh"
```
2. Restart terminal.
3. You should now be able to call `annonet` from any directory in the terminal.

### Choosing `model.h5`
By default, annonet will look for a compatible `model.h5` file for the necessary configurations to setup the internal model.
Path to a compatible model file can be provided to the argument `--modelpath` for a run to use the given model instead of the default one for the analysis.

> **`model.h5` file is not part of this repo, and you should ask the owners of the repo for it.**
>
> Once acquired, it is preferable that the file be placed as `$RENNET_ROOT/data/models/model.h5` (backup and replace if there is already such a file there).

### More help
```
$ ./annonet.sh -h
usage: rennet [-h] [--todir [TODIR]] [--modelpath [MODELPATH]] [--debug]
              [--version]
              infilepaths [infilepaths ...]

Annotate some audio files with rennet.

positional arguments:
  infilepaths           Paths to input audio files to be analyzed

optional arguments:
  -h, --help            show this help message and exit
  --todir [TODIR]       Path to output directory. Will be created if it
                        doesn't exist (default: respective directories of the
                        inputfiles)
  --modelpath [MODELPATH], -M [MODELPATH]
                        Path to the model file (default: $RENNET_ROOT/data/models/model.h5). Please add if
                        missing.
  --debug               Enable debugging mode where no errors are suppressed
                        during analysis.
  --version             show program's version number and exit
```

***

## Developers

### Python Library of Stuff that may be helpful in Speech Segmentation & Double Talk Detection

This is just the library of useful classes, functions, parsers, etc., and may be also application backend at some point.
`rennet/annonet.py` (being called by `annonet.sh`) provides the command line interface for applying a `rennet_model` on given speech files.
This `rennet_model` is stored in the `model.h5` discussed above.

For instructions on how to build an appropriate `rennet_model`, please refer to [`rennet-x`](https://bitbucket.org/nm-rennet/rennet-x).
It will be required to setup this repository in `dev-mode` before using [`rennet-x`](https://bitbucket.org/nm-rennet/rennet-x), so please follow the instructions below first.

### Setup in `dev-mode`

1. Make sure that `python` and `pip` are installed and available on your system.
    - Python 2.7 is probably pre-installed on Ubuntu and macOS. Sorry Windows user, but you'll have to figure some things out on your own.
    - Google on how to install `pip`.
    - This library is compatible with both Python 2.7.4+ and Python 3.5.3+, and you should maintain that, at least till 2020.
        + Actually, the target version is Python 3.5.3 at this point in time, but, to use the pre-installed versions on Ubuntu and macOS, support has been added for Python 2.7.4+ after-the-fact to minimize hassles for `annonet` users and to run on `nm-gpu-d`.

2. Clone this repository to your machine.
    - If you plan to run [`rennet-x`](https://bitbucket.org/nm-rennet/rennet-x) on `nm-gpu-d`, make sure that this repo is accessible from there _especially_ after `qsub -I` (e.g. clone into an appropriate location in `nm-raid`).

3. Setup and activate virtual environment with packages from `requirements/base.txt` and `requirements/dev.txt` installed.
    - With the terminal open at the root of this repository, running the following command will ask for, and then setup a Python virtual environment in `./.rennet_env`:
    ```
    ./annonet.sh
    ```
    - Activate the environment:
    ```
    source ./.rennet_env/bin/activate
    ```
    > **NOTE**
    >
    > This environment will not install the packages wtih GPU support.
    > Follow the instructions for that in [`rennet-x`](https://bitbucket.org/nm-rennet/rennet-x).

4. Install `base` and `dev` packages with `pip` by running the following:

        pip install -r requirements/base.txt
        pip install -r requirements/dev.txt
    

5. It is required that the environment variables `RENNET_ROOT` (pointing to the root of this repo) and `RENNET_ENV` (pointing to the virtual environment setup above) are available for working with [`rennet-x`](https://bitbucket.org/nm-rennet/rennet-x).
    - Add the following lines to your `~/.bash_profile`, for example:

            export RENNET_ROOT="path/to/local-rennet-repo"  # change here !!!
            export RENNET_ENV="$RENNET_ROOT/.rennet_env"

    - Reload `~/.bash_profile`, following the same example, to have these variables loaded, by running:
    ```
    source ~/.bash_profile
    ```

#### Check Installation

To check that everything has been setup properly, run the included tests using `pytest`. Since `rennet` has not been installed as a traditional Python package, you'll have to add it to the `PYTHONPATH` before calling `pytest`, as done in the commands below:

1. Change directory to the root of this repo:
    ```
    cd $RENNET_ROOT
    ```
2. Activate the virtual environment:
    ```
    source $RENNET_ENV/bin/activate
    ```
3. Add this library to `PYTHONPATH`:
    ```
    PYTHONPATH=$RENNET_ROOT:$PYTHONPATH
    ```
4. Run the tests in `tests`, excluding long running ones:
    ```
    py.test -rxs tests -m 'not long_running'
    ```

All tests _should_ pass if the installation was successful. If there are failures saying `cannot find rennet` or something similar, then the installation steps above have not been successful. If there are any other failures, get in touch with the owners of this repo, and may God help us.

### Debugging `annonet` Errors During Analysis

By default, the `annonet.sh` usage above will suppress all errors produced _during the analysis_ of a given speech file and move on to analyzing next file (errors may be thrown before or after that, e.g. due to an invalid `model.h5`).
To get the full stack-trace of the error, pass the additional flag `--debug` to `annonet.sh`.
Then, fix it.

For example:
```
$ ./annonet.sh some.WAV
Using Theano backend.

Analyzing 1/1 :
 /Users/USER/PATH/TO/some.WAV
272/272 [==============================] - 1442s     
rennet/annonet.py:95: RuntimeWarning: There was an error in analysing the given file:
(<type 'exceptions.KeyError'>,)
Moving to the next one.
  format(sys.exc_info()[:1])))

 DONE!
Output file can be found at:
```

```
$ ./annonet.sh some.WAV --debug
Using Theano backend.

Analyzing 1/1 :
 /Users/USER/PATH/TO/some.WAV
272/272 [==============================] - 1442s     
Traceback (most recent call last):
  File "rennet/annonet.py", line 87, in <module>
    outfiles.append(main(model, fp, to_dir=todir))
  File "rennet/annonet.py", line 39, in main
    return (rennet_model.apply(filepath, to_dir=to_dir))
  File "RENNET_ROOT/rennet/models.py", line 209, in apply
    self.output(x, to_filepath, audio_path=filepath)
  File "RENNET_ROOT/rennet/models.py", line 178, in output
    annotinfo_fn=self.seq_annotinfo_fn)
  File "RENNET_ROOT/rennet/utils/label_utils.py", line 704, in to_eaf
    eaf.add_linked_file(abspath(linked_media_filepath))
  File "RENNET_ENV/lib/python2.7/site-packages/pympi/Elan.py", line 314, in add_linked_file
    mimetype = self.MIMES[file_path.split('.')[-1]]
KeyError: 'WAV'
```

(Fixing this issue not shown)
