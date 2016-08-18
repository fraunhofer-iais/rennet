import pytest

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
