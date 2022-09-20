import logging
import numpy as np
import numba as nb
from itertools import groupby
from operator import itemgetter
from scipy.ndimage import binary_erosion
from snapdomen.imaging.dicomseries import DicomSeries
from typing import Tuple, Dict


@nb.njit
def _convert_to_fat_fraction(hu: float) -> float:
    """
    convert a hounsfield unit to fat fraction
    :param hu: the hounsfield unit measurement to convert
    :return: fat fraction value
    """
    ff = -0.58 * hu + 38.2
    return ff


def longest_consecutive_seg(numbers):
    """
    A function to find the longest consecutive cut of liver from a segmentation
    :param numbers: a list of indices containing liver segmentations
    :return: the starting and ending index of the longest consecutive cut
    """
    idx = max(
        (
            list(map(itemgetter(0), g))
            for i, g in groupby(enumerate(np.diff(numbers) == 1), itemgetter(1))
            if i
        ),
        key=len
    )
    return idx[0], idx[-1] + 1


def calculate_pixel_radius(series: DicomSeries, roi_area):
    """
    Calculate the radius of an ROI in pixels
    :param series: the dicom series of interest
    :param roi_area: desired area of ROI in centimeters
    :return: radius in pixels for the ROI
    """
    # Get the pixel width in centimeters
    pixel_width = series.spacing[0] / 10
    radius = np.sqrt(roi_area / np.pi)
    pixel_radius = int(np.ceil(radius / pixel_width))
    return pixel_radius


def create_roi(center: tuple, radius: int, h: int = 512, w: int = 512) -> np.ndarray:
    """
    A function to create circular ROIs of set radii around points of interest in liver \n
    :param center: the point of interest to draw a circle around (in x,y coordinates)
    :param radius: length of the radius around the center
    :param h: height of the ct image in pixels (default: 512)
    :param w: width of the ct image in pixels (default: 512)
    :return: a boolean numpy array containing the circular ROI
    """
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    circle = dist_from_center <= radius
    return circle


def prepare_seg_output(segmentation: np.ndarray, threshold: int = 0):
    """
    Postprocess the organ segmentation from the UNet
    :param segmentation: the output from the UNet
    :param threshold: the minimum number of pixels to be considered part of the organ (default: 0)
    :return: a boolean array of the organ segmentation
    """
    seg = np.squeeze(segmentation)
    seg = np.round(seg)
    # Get the longest consecutive segmentation of the liver
    has_organ = []
    for i in range(seg.shape[0]):
        val = seg[i, :, :].sum()
        if val > threshold:
            has_organ.append(i)
    start, end = longest_consecutive_seg(has_organ)
    # Create the final segmentation
    # Set any slices that don't have liver to zero
    seg[:start, :, :] = 0
    seg[end:, :, :] = 0
    return seg


def get_midpoint(pt1: tuple, pt2: tuple) -> tuple:
    """
    A function to find the midpoint between two points
    :param pt1: first point
    :param pt2: second point
    :return: the midpoint between the two points
    """
    return int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)


def find_center_pt(seg_iso: np.ndarray, single: np.ndarray, direction: str, pixel_radius: float, use_alpha: bool, alpha: float) -> tuple:
    """
    A function to find the center of an ROI in a specific direction\n
    :param seg_iso: the center slice of a segmentation derived from erode3d
    :param single: the erosion center point derived from erode3d
    :param direction: Direction to draw ROI, accepted vals are ('up', 'down', 'left', 'top_left', and 'bot_left')
    :param pixel_radius: the radius of the ROI, will draw it 2*radius pixels away from edge
    :return: a tuple containing the (x, y) coordinates to draw the ROI around
    """

    # Define the center points
    x_center = single[1]
    y_center = single[0]
    start = np.inf
    center = False
    pixel_diameter = 2 * pixel_radius

    if direction == 'left':
        # Find the edge to the left of the center point
        for i in range(seg_iso.shape[1]):
            if seg_iso[y_center][i] == 1:
                start = i
                break
        # Find the roi center point
        if use_alpha:
            center = (start + (x_center - start) * alpha, y_center)
        else:
            center = (start + pixel_diameter, y_center)

    elif direction == 'up':
        # Find the edge above the center point
        for i in range(seg_iso.shape[0]):
            if seg_iso[i][x_center] == 1:
                start = i
                break
        if use_alpha:
            center = (x_center, start + (y_center - start) * alpha)
        else:
            center = (x_center, start + pixel_diameter)

    elif direction == 'down':
        # Find the edge below the center point
        for i in range(seg_iso.shape[0]):
            if seg_iso[seg_iso.shape[0] - 1 - i][x_center] == 1:
                start = seg_iso.shape[0] - 1 - i
                break
        if use_alpha:
            center = (x_center, start - (start - y_center) * alpha)
        else:
            center = (x_center, start - pixel_diameter)

    elif direction == 'top_left':
        # Find the edge at approximately 120 degrees
        for i in range(seg_iso.shape[0]):
            if seg_iso[y_center - 2 * i][x_center - i] == 0:
                center = (x_center - i - 1 + pixel_diameter, y_center + 2 * (i - 1) - pixel_diameter)
                break

    elif direction == 'bot_left':
        for i in range(seg_iso.shape[0]):
            if seg_iso[y_center + 2 * i][x_center - i] == 0:
                center = (x_center - i - 1 + pixel_diameter, y_center + 2 * (i - 1) - pixel_diameter)
                break

    else:
        center = None

    if not center:
        raise ValueError('Calculation broke somewhere, center point is None')

    return center


def find_center_pt(seg_iso: np.ndarray, single: np.ndarray, direction: str, pixel_radius: float, use_alpha: bool, alpha: float) -> tuple:
    """
    A function to find the center of an ROI in a specific direction\n
    :param seg_iso: the center slice of a segmentation derived from erode3d
    :param single: the erosion center point derived from erode3d
    :param direction: Direction to draw ROI, accepted vals are ('up', 'down', 'left', 'top_left', and 'bot_left')
    :param pixel_radius: the radius of the ROI, will draw it 2*radius pixels away from edge
    :return: a tuple containing the (x, y) coordinates to draw the ROI around
    """

    # Define the center points
    x_center = single[1]
    y_center = single[0]
    start = np.inf
    center = False
    pixel_diameter = 2 * pixel_radius

    if direction == 'left':
        # Find the edge to the left of the center point
        for i in range(seg_iso.shape[1]):
            if seg_iso[y_center][i] == 1:
                start = i
                break
        # Find the roi center point
        if use_alpha:
            center = (start + (x_center - start) * alpha, y_center)
        else:
            center = (start + pixel_diameter, y_center)

    elif direction == 'up':
        # Find the edge above the center point
        for i in range(seg_iso.shape[0]):
            if seg_iso[i][x_center] == 1:
                start = i
                break
        if use_alpha:
            center = (x_center, start + (y_center - start) * alpha)
        else:
            center = (x_center, start + pixel_diameter)

    elif direction == 'down':
        # Find the edge below the center point
        for i in range(seg_iso.shape[0]):
            if seg_iso[seg_iso.shape[0] - 1 - i][x_center] == 1:
                start = seg_iso.shape[0] - 1 - i
                break
        if use_alpha:
            center = (x_center, start - (start - y_center) * alpha)
        else:
            center = (x_center, start - pixel_diameter)

    elif direction == 'top_left':
        # Find the edge at approximately 120 degrees
        for i in range(seg_iso.shape[0]):
            if seg_iso[y_center - 2 * i][x_center - i] == 0:
                center = (x_center - i - 1 + pixel_diameter, y_center + 2 * (i - 1) - pixel_diameter)
                break

    elif direction == 'bot_left':
        for i in range(seg_iso.shape[0]):
            if seg_iso[y_center + 2 * i][x_center - i] == 0:
                center = (x_center - i - 1 + pixel_diameter, y_center + 2 * (i - 1) - pixel_diameter)
                break

    else:
        center = None

    if not center:
        raise ValueError('Calculation broke somewhere, center point is None')

    return center


def erode3d(seg: np.ndarray) -> np.ndarray:
    """
    A function to find the erosion of a 3d segmentation of liver\n
    :param seg: a numpy array holding the whole liver segmentation
    :return: a numpy array containing the erosion of the segmentation
    """
    # Create the starting kernel
    kernel = np.zeros([3, 3, 3])
    kernel[1, 1, 0] = 1
    kernel[0, 1, 1] = 1
    kernel[1, 0, 1] = 1
    kernel[1, 1, 1] = 1
    kernel[1, 2, 1] = 1
    kernel[2, 1, 1] = 1
    kernel[1, 1, 2] = 1
    # Erode the mask down to a center point (y, x, z)
    eroded = binary_erosion(seg, structure=kernel).astype(seg.dtype)
    return eroded


def erode_seg(seg_erode: np.ndarray, erosion_limit=100) -> np.ndarray:
    """
    A function to erode down a segmentation using erode3d \n
    :param seg_erode: segmentation of original size to erode
    :param erosion_limit: limit to stop eroding the image down (default: 100)
    :return: a numpy array containing the center point
    """
    i = 0
    while np.sum(seg_erode) > erosion_limit:
        erode_last = seg_erode
        seg_erode = erode3d(seg_erode)
        i += 1

    if seg_erode.sum() == 0:
        seg_erode = erode_last

    single = np.argwhere(seg_erode)
    single = single[int(len(single) / 2)]
    return single


def create_roi_masks(seg_iso: np.ndarray, single: np.ndarray, pixel_radius: int, use_alpha: bool, alpha: float, n_roi:int) -> np.ndarray:
    """
    A function to draw ROIs in appropriate locations based on a liver segmentation \n
    :param seg_iso: center slice of the segmentation (from isolate_center_slice)
    :param single: the center point of the liver segmentation (y, x, z)
    :param pixel_radius: the radius in pixels for the desired ROIs
    :param use_alpha: whether to use the alpha value to determine the distance from the edge
    :param alpha: the distance away from the edge to draw the ROIs relative to the center
    :param n_roi: the number of ROIs to draw
    :return: a new mask containing 3 ROIs that should be contained within the liver
    """

    height, width = seg_iso.shape
    new_mask = np.zeros(seg_iso.shape)

    if n_roi == 1:
        # If we only want one ROI, we can just use the center point of the erosion and make the roi larger
        center = (single[1], single[0])
        logging.info(f'Drawing ROI at {center}')
        roi = create_roi(center=center, radius=pixel_radius, h=height, w=width)
        new_mask[roi] = 1
        return new_mask

    if n_roi == 3 or n_roi == 5:
        directions = ['up', 'down', 'left']
        logging.info(f'Using Alpha Value: {use_alpha}')
        for value, direction in enumerate(directions):
            center = find_center_pt(seg_iso=seg_iso, single=single, direction=direction, pixel_radius=pixel_radius,
                                    use_alpha=use_alpha, alpha=alpha)
            logging.info(f'Drawing ROI {value + 1} at {center}')
            roi = create_roi(center=center, radius=pixel_radius, h=height, w=width)
            new_mask[roi] = value + 1
            if direction == 'left':
                left_center = center
            elif direction == 'up':
                top_center = center
            elif direction == 'down':
                bot_center = center

        if n_roi == 5:
            for value, direction in enumerate(['top_left', 'bot_left']):
                if direction == 'top_left':
                    center = get_midpoint(left_center, top_center)
                elif direction == 'bot_left':
                    center = get_midpoint(left_center, bot_center)

                logging.info(f'Drawing ROI {value + 4} at {center}')
                roi = create_roi(center=center, radius=pixel_radius, h=height, w=width)
                new_mask[roi] = value + 4

        return new_mask

    else:
        raise ValueError('Number of ROIs must be 1, 3 or 5')


def _measure_fat(series: DicomSeries, segmentation: np.ndarray, single_slice_loc=None) -> Tuple:
    """
    Measure the hounsfield units/fat fraction from each
    :param series: the dicom series
    :param segmentation: the organ segmentation that has been preprocessed
    :return: a tuple containing the average hounsfield units and fat fraction of the volume of the segmentation
    """
    original_image = series.pixel_array.copy()
    if single_slice_loc is not None:
        original_image = original_image[single_slice_loc]
    liver_indices = np.where(segmentation == 1)
    hounsfield_units = original_image[liver_indices]
    average_hu = hounsfield_units.mean()
    average_fat_fraction = _convert_to_fat_fraction(average_hu)
    return average_hu, average_fat_fraction


def measure_fat_in_tissue(series: DicomSeries, segmentation: np.ndarray, organ: str, pixel_radius: int, use_alpha: bool, alpha: float, n_roi: int) -> Dict:
    """
    A function to measure the fat from organ tissue segmentation
    :param series: the original dicom series
    :param segmentation: the output from the UNet segmentation
    :param organ: the organ being measured
    :return: a dictionary containing the tissue fat information
    """
    # original_scan = series.pixel_array.copy()
    info = {}
    if organ == 'spleen':
        threshold = 20
    else:
        threshold = 0
    seg = prepare_seg_output(segmentation, threshold)
    # Volume measurements
    average_volume_hu, average_volume_fat = _measure_fat(series, seg)
    info[f'{organ}_volume_hu'] = average_volume_hu
    info[f'{organ}_volume_fat'] = average_volume_fat

    # Slice measurements
    center_idx = erode_seg(seg, erosion_limit=100)
    # center_image = series.pixel_array[center_idx[0]].copy()
    center_seg = seg[center_idx[0]].copy()
    average_slice_hu, average_slice_fat = _measure_fat(series, center_seg, single_slice_loc=center_idx[0])
    info[f'{organ}_center_hu'] = average_slice_hu
    info[f'{organ}_center_fat'] = average_slice_fat

    # ROI Measurements
    # roi_mask = create_roi_masks(center_seg, center_idx, pixel_radius, use_alpha, alpha, n_roi)

    return info
