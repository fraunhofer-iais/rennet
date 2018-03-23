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
"""Tests for sanity of installed packages."""
import pytest
from rennet.utils.audio_utils import get_codec

# pylint: disable=unused-variable


@pytest.mark.trivial
@pytest.mark.sanity_check
def test_numpy_installed():
    import numpy
    assert True


@pytest.mark.trivial
@pytest.mark.sanity_check
def test_scipy_installed():
    import scipy
    assert True


@pytest.mark.trivial
@pytest.mark.sanity_check
def test_tensorflow_installed():
    import tensorflow
    assert True


@pytest.mark.trivial
@pytest.mark.sanity_check
def test_pydub_installed():
    import pydub
    assert True


@pytest.mark.trivial
@pytest.mark.sanity_check
def test_codec_available():
    assert get_codec()
