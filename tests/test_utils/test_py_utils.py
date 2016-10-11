"""
@motjuste
Created: 10-10-2016

Test the Pure Python utilities
"""
from rennet.utils import py_utils as pu


def test_cvsecs():
    """ Test cvsecs that converts time-string to seconds """
    assert pu.cvsecs(15.4) == 15.4  # seconds
    assert pu.cvsecs((1, 21.5)) == 81.5  # (min,sec)
    assert pu.cvsecs((1, 1, 2)) == 3662  # (hr, min, sec)
    assert pu.cvsecs('01:01:33.5') == 3693.5  #(hr,min,sec)
    assert pu.cvsecs('01:01:33.045') == 3693.045
    assert pu.cvsecs('01:01:33,5') == 3693.5  #coma works too
