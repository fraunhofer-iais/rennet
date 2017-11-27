# RENNET

> *ˈrɛnɪt*
>
> curdled milk from the stomach of an unweaned calf, containing rennin and used in curdling milk for cheese.

## Usage

1. In the terminal, change to the root of the local copy of this repository.
2. Run the following to analyze the two files, for example.
```
./annonet.sh path/to/file1.wav path/to/file2.mp4
```

> The first time the command is run, a local environment will be setup, which requires `git`, a connection to the internet and some time. Subsequent runs will not require this setup.

### Creating an Alias
1. Add the following to your `~/.bash_profile`, with the appropriate `path/to/local-rennet-repo`:
```
export RENNET_ROOT="path/to/local-rennet-repo"
alias annonet="./$RENNET_ROOT/annonet.sh"
```
2. Restart terminal.
3. You should now be able to call `annonet` from any directory in the terminal.

### Choosing `model.h5`
By default, annonet will look for a compatible `model.h5` file for the necessary configurations to setup the internal model.
Path to a compatible model file can be provided to the argument `--modelpath` for a run to use the given model instead of the default one for the analysis.

> **`model.h5` file is not part of this repo, and you should ask the owners of the repo for it.**
>
> Once acquired, it is preferable that the file be placed as `./data/models/model.h5` (backup and replace if there is already such a file there).

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

## Developers
### Library of Stuff that may be helpful in Speech Segmentation & Double Talk Detection

This is just the library of useful classes, functions, parsers, etc., and may be also application backend at some point.

Preferred version of Python is 3.5, may be soon 3.6. Using with Python 2.7 hasn't failed yet though.

For examples of usage in real experiments and experiences, please refer to [rennet-x](https://bitbucket.org/nm-rennet/rennet-x).

Until this library/package does not have a proper `setup.py`, an easy way to use it would be to add the root of the local copy of this repository to your `PYTHONPATH` variable.
Then you will be able to access the stuff in this library as any other Python library's stuff.
Refer to [rennet-x](https://bitbucket.org/nm-rennet/rennet-x) for inspirations.

It is likely that many things will not be working as is in [rennet-x](https://bitbucket.org/nm-rennet/rennet-x).

That repo, and this, are results of a great divorce, so some things may be unexpectedly broken. Please do shout out if you find something.
Be patient until the APIs are finalized.
