"""
@motjuste
Craated: 09-11-2017

The command line interface for rennet, called annonet.
"""
from __future__ import print_function
import os
import sys
try:
    rennet_root = os.environ['RENNET_ROOT']
except KeyError:
    raise RuntimeError("Environment variable RENNET_ROOT is not set.")
import argparse
from h5py import File as hf
import warnings

from rennet import __version__ as currver
import rennet.utils.model_utils as mu
import rennet.models as m

DEFAULT_MODEL_PATH = os.path.join(rennet_root, "data", "models", "model.h5")


def validate_and_init_rennet_model(model_fp):
    try:
        with hf(model_fp, 'r') as f:
            minver = f['rennet'].attrs['version_min']
            srcver = f['rennet'].attrs['version_src']
            modelname = f['rennet/model'].attrs['name']

        mu.validate_rennet_version(minver, srcver)
        return m.get(modelname)(model_fp)
    except KeyError:
        raise RuntimeError("Invalid model file: {}".format(model_fp))


def main(rennet_model, filepath, to_dir=None):
    return (rennet_model.apply(filepath, to_dir=to_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Annotate some audio files with rennet.", prog='rennet'
    )

    parser.add_argument(
        'infilepaths',
        nargs='+',
        type=argparse.FileType('r'),
        help="Paths to input audio files to be analyzed",
    )
    parser.add_argument(
        '--todir',
        nargs='?',
        help=
        "Path to output directory. Will be created if it doesn't exist (default: respective directories of the inputfiles)",
        default=None,
    )
    parser.add_argument(
        '--modelpath',
        '-M',
        nargs='?',
        type=argparse.FileType('r'),
        default=DEFAULT_MODEL_PATH,
        help="Path to the model file \n(default: {}).\nPlease add if missing.".
        format(DEFAULT_MODEL_PATH),
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debugging mode where no errors are suppressed during analysis."
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {}'.format(currver),
    )
    args = parser.parse_args()

    modelfp = args.modelpath.name
    model = validate_and_init_rennet_model(modelfp)
    model.verbose = 1

    outfiles = []
    absinfilepaths = list(map(os.path.abspath, (f.name for f in args.infilepaths)))
    total_files = len(absinfilepaths)

    todir = os.path.abspath(args.todir) if args.todir is not None else None
    debug_mode = args.debug
    for i, fp in enumerate(absinfilepaths):
        print("\nAnalyzing {}/{} :\n".format(i + 1, total_files), fp)
        try:
            outfiles.append(main(model, fp, to_dir=todir))
            print("Output created at", outfiles[-1])
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # pylint: disable=bare-except

            if debug_mode:
                raise
            else:
                # NOTE: Catch all for errors so that one mis-behaving file doesn't mess all of them
                warnings.warn(
                    RuntimeWarning(
                        "There was an error in analysing the given file:\n{}\n".format(sys.exc_info()[:1])+\
                        "Add '--debug' at the end of your call to annonet to get a full stacktrace.\n"+\
                        "Moving to the next audio."
                        ))

    print(
        "\n DONE!",
        "Output file{} can be found at:".format("s" if len(outfiles) > 1 else ""),
        *outfiles,
        sep='\n'
    )
