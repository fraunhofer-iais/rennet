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
