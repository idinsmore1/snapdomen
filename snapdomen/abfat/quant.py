import numpy as np

from scipy.ndimage import binary_fill_holes, binary_erosion
from .model import build_unet

from snapdomen.imaging.utils import normalize_image
from snapdomen.waistcirc.measure import get_largest_connected_component, get_waist_circumference


def extract_abdomen(pixel_array, start_point, end_point):
    """
    Only extract the slices that are included in the abdomen
    :param pixel_array: the pixel data that make up the ct scan
    :param start_point: start point axial index (usually the l1 vertebra)
    :param end_point: end point axial index (usually the l5 vertebra)
    :return: the extracted abdomen array
    """
    abdomen = pixel_array[start_point: end_point, : , :].copy()
    return abdomen


def postprocess_prediciton(preds):
    """
    process prediction image from abdomen segmentation for fat quantification\n
    :param preds: the prediction output
    :return: the processed predictions
    """
    preds = np.squeeze(preds)
    preds = np.round(preds)
    new_preds = np.zeros_like(preds)
    for i in range(len(preds)):
        new_pred = get_largest_connected_component(preds[i])
        new_pred = binary_fill_holes(new_pred)
        new_pred = binary_erosion(new_pred)
        new_preds[i] = new_pred
    return new_preds


def separate_abdominal_cavity(image, abd_pred):
    """
    Get the interior and exterior abdominal masks using the postprocessed prediction from unet
    :param image: the original axial image slice from the ct scan
    :param abd_pred: the postprocessed abdominal segmentation for the slice
    :return: the interior and exterior abdominal masks
    """
    interior = np.ma.masked_where(abd_pred == 1, image)
    interior = np.ma.getmask(interior)

    exterior = np.ma.masked_where(abd_pred == 0, image)
    exterior = np.ma.getmask(exterior)

    return interior, exterior


def measure_fat(image, mask, pixel_height, pixel_width, window=(-190, -30)):
    copied = image.copy()
    copied[mask == False] = np.min(image)
    fat_pixels = ((copied > window[0]) & (copied < window[1])).sum()
    fat_area = fat_pixels * pixel_height * pixel_width
    return fat_pixels, fat_area

