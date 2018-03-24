#  Copyright 2018 Fraunhofer IAIS. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""The command line interface for rennet, called annonet.

@motjuste
Craated: 09-11-2017
"""
from __future__ import print_function
import argparse
import os
import sys
import warnings
from h5py import File as hf

from rennet import __version__ as currver
import rennet.utils.model_utils as mu
import rennet.models as m

RENNET_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(RENNET_ROOT, "data", "models", "model.h5")


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
    return rennet_model.apply(filepath, to_dir=to_dir)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description="Annotate some audio files with rennet.", prog='rennet'
    )

    PARSER.add_argument(
        'infilepaths',
        nargs='+',
        type=argparse.FileType('r'),
        help="Paths to input audio files to be analyzed",
    )
    PARSER.add_argument(
        '--todir',
        nargs='?',
        help=(
            "Path to output directory. Will be created if it doesn't exist " +
            "(default: respective directories of the inputfiles)"
        ),
        default=None,
    )
    PARSER.add_argument(
        '--modelpath',
        '-M',
        nargs='?',
        type=argparse.FileType('r'),
        default=DEFAULT_MODEL_PATH,
        help="Path to the model file \n(default: {}).\nPlease add if missing.".
        format(DEFAULT_MODEL_PATH),
    )
    PARSER.add_argument(
        '--debug',
        action='store_true',
        help="Enable debugging mode where no errors are suppressed during analysis."
    )
    PARSER.add_argument(
        '--version',
        action='version',
        version='%(prog)s {}'.format(currver),
    )
    # pylint: disable=invalid-name
    args = PARSER.parse_args()

    modelfp = args.modelpath.name
    model = validate_and_init_rennet_model(modelfp)
    model.verbose = 1

    outfiles = []
    absinfilepaths = [os.path.abspath(f.name) for f in args.infilepaths]
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
                msg = "There was an error in analysing the given file:\n{}\n".format(
                    sys.exc_info()[:1]
                )
                msg += "Pass the '--debug' flag to annonet to get a full stacktrace.\n"
                msg += "Moving to the next audio."
                warnings.warn(RuntimeWarning(msg))

    print(
        "\n DONE!",
        "Output file{} can be found at:".format("s" if len(outfiles) > 1 else ""),
        *outfiles,
        sep='\n'
    )
    # pylint: enable=invalid-name
