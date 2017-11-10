"""
@motjuste
Craated: 09-11-2017

The command line interface for rennet, called annonet.
"""
from __future__ import print_function
import os
try:
    rennet_root = os.environ['RENNET_ROOT']
except KeyError:
    raise RuntimeError("Environment variable RENNET_ROOT is not set.")
import argparse
from h5py import File as hf

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
        description="Annotate some audio files with rennet.", prog='rennet')

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
        "Path to output directory. Will be created if it doesn't exist (default: respective directory of the inputfiles)",
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
        '--version',
        action='version',
        version='%(prog)s {}'.format(currver),
    )
    args = parser.parse_args()

    modelfp = args.modelpath.name
    model = validate_and_init_rennet_model(modelfp)
    model.verbose = 1

    outfiles = []
    for fp in args.infilepaths:
        print("\nAnalyzing", fp.name)
        outfiles.append(main(model, fp.name, to_dir=args.todir))

    print(
        "\n DONE!",
        "Output file{} can be found at:".format("s"
                                                if len(outfiles) > 1 else ""),
        *outfiles,
        sep='\n')
