"""
@motjuste
Created: 10-10-2016

Pure Python utilities
"""
import re


def is_string(obj):
    """ Returns true if s is string or string-like object,
    compatible with Python 2 and Python 3.
    """
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)


def cvsecs(time):
    """ Will convert any time into seconds.

    Here are the accepted formats:

    >>> cvsecs(15.4) -> 15.4 # seconds
    >>> cvsecs( (1,21.5) ) -> 81.5 # (min,sec)
    >>> cvsecs( (1,1,2) ) -> 3662 # (hr, min, sec)
    >>> cvsecs('01:01:33.5') -> 3693.5  #(hr,min,sec)
    >>> cvsecs('01:01:33.045') -> 3693.045
    >>> cvsecs('01:01:33,5') #coma works too

    """
    if is_string(time):
        if (',' not in time) and ('.' not in time):
            time = time + '.0'
        expr = r"(\d+):(\d+):(\d+)[,|.](\d+)"
        finds = re.findall(expr, time)[0]
        nums = [float(f) for f in finds]
        return (3600 * int(finds[0]) + 60 * int(finds[1]) + int(finds[2]) +
                nums[3] / (10**len(finds[3])))

    elif isinstance(time, tuple):
        if len(time) == 3:
            hr, mn, sec = time
        elif len(time) == 2:
            hr, mn, sec = 0, time[0], time[1]
        return 3600 * hr + 60 * mn + sec

    else:
        return time


class BaseSlotsOnlyClass(object):  #pylint: disable=too-few-public-methods
    """ Slots only base class.

    Implements creating repr automatically.
    """
    # TODO: [ ] Dox
    # TODO: [ ] Tests!

    __slots__ = ()

    def __repr__(self):
        a_v = ((att, getattr(self, att)) for att in self.__slots__)
        r = ".".join((self.__module__.split(".")[-1], self.__class__.__name__))
        return r + "({})".format(", ".join("{!s}={!r}".format(*av) for av in a_v))
