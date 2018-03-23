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
"""Test the Pure Python utilities

@motjuste
Created: 10-10-2016
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
