from __future__ import print_function
import time

print()
print("/ {:/<80}".format(str(time.asctime()) + " "))

import sys
print("\nPython Version:", sys.version)

try:
    import tensorflow as tg
    print("\nTesnorflow Version:", tg.__version__)
except:
    print("\nProblem Loading tensorflow")

try:
    import six
    print("\nCan Load six")
except:
    print("\nProblem loading six")

import rennet.utils.audio_utils as au
print("\nCan Load rennet\n")
