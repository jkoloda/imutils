"""Various decorators for common image processing.

Note
----
Decorators must be used without (explicit) parameters. If we want to
change the default parameter values, perhaps the easiest thing is to create
an auxiliary decorator using functool.partial. For instance:

from functools import partial
custom_imclip = partial(imclip, minval=-100, maxval=100)

@custom_imclip
def func():
    ...

"""

import numpy as np


def imclip(func, minval=0, maxval=1):
    """Clip output image to fit within [minval, maxval] interval."""
    def wrapper(*args, **kwargs):
        img = func(*args, **kwargs)
        img[img < minval] = minval
        img[img > maxval] = maxval
        return img

    return wrapper


def imrange(func, minval=0, maxval=1):
    """Check input image range to be within specified interval boundaries.

    Note
    ----
    It is assumed that image array is the first parameter
    passed to the decorated function.
    """
    def wrapper(img, *args, **kwargs):
        assert (np.min(img) >= minval and np.max(img) <= maxval)
        img = func(img, *args, **kwargs)
        return img

    return wrapper
