import cv2
import numpy as np
from imutils.decorators import imclip, imrange


@imclip
@imrange
def imnoise(img, std=0.1, random=False):
    """Add gaussian noise with zero mean (AWGN) to input image.

    Parameters
    ----------
    img : ndarray
        Input image in range [0, 1].

    random : bool
        If True, random std uniformly sampled from interval [0, std]
        is applied.

    Returns
    -------
    img : ndarray
        Image with added noise.

    """
    rows, cols, dims = img.shape
    if random is True:
        std = std * random_number(sign=False)
    noise = std * np.random.randn(rows, cols, dims)
    img = img + noise
    return img


@imclip
@imrange
def imbright(img, shift=0.1, random=False):
    """Shift all pixel values (modify brightness) by a certain amount.

    Parameters
    ----------
    img : ndarray
        Input image in the range [0, 1].

    shift : float
        Amount of shift between 0 and 1. Positive shifts make image brighter,
        negative shifts make image darker.

    random : bool
        If True, random shift uniformly sampled from interval [-shift, +shift]
        is applied.

    Returns
    -------
    img : ndarray
        Image with modified brightness.

    """
    if random is False:
        img = img + shift
    else:
        img = img + shift * random_number(sign=True)

    return img


def imflip(img, random=False):
    """ Horizontal flip. TODO:_ vertical"""
    flip = random_number() > 0.5
    if flip is False:
        return img
    else:
        if not isinstance(img, list):
            img = [img]
        img = [np.fliplr(i) for i in img]


@imclip
@imrange
def imgamma(img, gamma, random=False):
    """Perform gamma correction on image.

    Parameters
    ----------
    img : ndarray
        Input image in the range [0, 1].

    gamma : float
        Value for gamma correction. Values smaller than 1 brighten input images
        (recall taht input image range is [0, 1]). Values larger than 1 darken
        input images.

    random : bool
        If True, random gamma uniformly sampled from interval [1/gamma, gamma]
        is applied.

    Returns
    -------
    img : ndarray
        Image with gamma correction applied.

    """
    assert gamma > 0
    if random is False:
        return np.power(img, gamma)
    else:
        # Generate random value between 1 and gamma (+ or -)
        gamma = max(gamma, 1/gamma)
        delta = (gamma - 1) * random_number(sign=True)
        if delta > 0:
            gamma = 1 + delta
        else:
            # Negative values are mapped to interval [1/gamma, 1]
            gamma = 1.0 / (1 + abs(delta))
        return np.power(img, gamma)


def random_number(sign=False):
    """Return random number between 0 and 1 either signed or unsigned.

    Parameters
    ----------
    sign : bool
        If True, random number from interval [0, 1] is returned. Otherwise
        random number from [-1, 1] is returned.

    Returns
    -------
    float
        Random number between 0 and 1 (signed or unsigned).

    """
    if sign is False:
        return np.random.random()
    else:
        return (np.random.random() - 0.5) * 2


def imcrop(img, size):
    """Randomly crop image to fit specified size.

    Parameters
    ----------
    img : ndarray
        Image to be cropped.

    size : tuple
        Resulting size of image after crop, given as (rows, cols).

    Returns
    -------
    img : ndarray
        Cropped image.

    """
    assert img.shape[0] >= size[0] and img.shape[1] >= size[1]

    # Compute maximum possible offsets (horizontal and vertical)
    max_offset_row = img.shape[0] - size[0]
    max_offset_col = img.shape[1] - size[1]

    # Generate random offsets to be applied during crop
    offset_row = np.random.randint(0, max_offset_row + 1)
    offset_col = np.random.randint(0, max_offset_col + 1)

    # Apply crop
    return img[offset_row:offset_row + size[0],
               offset_col:offset_col + size[1],
               ...]


def imread(filename, standardize=False):
    """Read image as *RGB* with values between 0 and 1.

    Parameters
    ----------
    filename : str
        Filename of image to load.

    standardize : bool
        Indicates whether to standardize image, i.e., to subtract mean value
        and divide by standard deviation.

    Returns
    -------
    ndarray
        Loaded image.

    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.0
    if standardize is True:
        return imstandardize(img)
    else:
        return img


def imstandardize(img):
    """Standardize image by subtracting mean and dividing by std.

    Parameters
    ----------
    img : ndarray
        Input image to standardize

    Returns
    -------
    img : ndarray
        Standardize input image

    """
    img = img.astype(np.float32)
    img = (img - np.mean(img))/np.std(img)
    return img
