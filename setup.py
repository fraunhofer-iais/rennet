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
"""A setuptools based setup module for rennet
@motjuste
Created: 19-03-2018

ref:
    - https://packaging.python.org/en/latest/distributing.html
    - https://github.com/pypa/sampleproject
"""
from setuptools import setup, find_packages
from codecs import open as copen
from os import path

HERE = path.abspath(path.dirname(__file__))

# Get the version from rennet/version.py
_version_globals = {}
with copen(path.join(HERE, "rennet", "version.py"), 'r', 'utf-8') as vf:
    exec(vf.read(), _version_globals)  # pylint: disable=exec-used

VERSION = _version_globals.get("VERSION", "SOMETHING.TERRIBLY.WRONG")

# Use README.md as long description
with copen(path.join(HERE, "README.md"), 'r', 'utf-8') as rf:
    LONG_DESCRIPTION = rf.read()
    LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

######################################################################### REQUIREMENTS #
INSTALL_REQUIRES = [
    "six",  # your days are numbered anyways
    "numpy >= 1.12.1",
    "matplotlib >= 2.0.2",
    "librosa >= 0.5.0",
    "h5py >= 2.7.0",
    "tensorflow >= 1.0.0",
    "keras >= 2.0.2",
    "pydub >= 0.18.0",
    "pympi-ling >= 1.69",
]

ANALYSIS_REQUIRES = [
    "jupyter >= 1.0.0",
    "dask[array, distributed] >= 0.17.0",
    "tqdm >= 4.11.2",
]

TEST_REQUIRES = [
    "pytest-watch",  # implicitly install pytest
    "pytest-cov",
    "pytest-pylint",  # implicitly installs pylint
]

DEV_REQUIRES = TEST_REQUIRES + [
    "yapf",
    "pytest-benchmark",
    "line-profiler",
]

################################################################################ SETUP #
setup_params = {
    "name": "rennet",
    "version": VERSION,
    "description": "Deep Learning utilities, mainly for audio segmentation ... for now",
    "long_description_content_type": LONG_DESCRIPTION_CONTENT_TYPE,
    "long_description": LONG_DESCRIPTION,
    "author": "Fraunhofer IAIS",
    "license": "Apache License 2.0",
    "url": "https://github.com/fraunhofer-iais/rennet",
    "packages": find_packages(),
    "install_requires": INSTALL_REQUIRES,
    "extras_require": {
        "analysis": ANALYSIS_REQUIRES,
        "test": TEST_REQUIRES,
        "dev": DEV_REQUIRES,
    },
    # TODO: Add console-script for annonet, or a better replacement
    # TODO: Add PyPI classifiers and keywords
}

if __name__ == '__main__':
    setup(**setup_params)
