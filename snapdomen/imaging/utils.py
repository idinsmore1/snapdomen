import numpy as np
from numba import njit


def normalize_image(img):
    """ Normalize image values to [0,1] \n
    :param img: the image to be normalized
    :return: the normalized image
    """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)


@njit
def transform_to_hu(image: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """
    A function to transform a ct scan image into hounsfield units \n
    :param image: a numpy array containing the raw ct scan
    :param slope: the slope rescaling factor from the dicom file (0 if already in hounsfield units)
    :param intercept: the intercept value from the dicom file (depends on the machine)
    :return: a copy of the numpy array converted into hounsfield units
    """
    hu_image = image * slope + intercept
    return hu_image


@njit
def window_image(image: np.ndarray, window_center: int, window_width: int):
    """
    A function to window the hounsfield units of the ct scan \n
    :param image: a numpy array containing the hounsfield ct scan
    :param window_center: hounsfield window center
    :param window_width: hounsfield window width
    :return: a windowed copy of 'image' parameter
    """
    # Get the min/max hounsfield units for the dicom image
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    new_image = np.clip(image, img_min, img_max)
    return new_image
