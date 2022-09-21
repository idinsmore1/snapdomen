import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

from numba import njit
from numba.typed import List
from skimage.draw import disk
from typing import Tuple, Dict

from snapdomen.abfat.model import build_unet
from snapdomen.imaging.dicomseries import DicomSeries
from snapdomen.imaging.utils import normalize_image
from snapdomen.imaging.organseg import OrganSeg


@njit
def find_roi_centers(mask, mask_center, alpha=0.55):
    # Get the center coordinates
    y_center = mask_center[1]
    x_center = mask_center[2]
    antialpha = 1 - alpha
    # Get the edges
    left_edge = np.where(mask[y_center, :] == 1)[0].min()
    top_edge = np.where(mask[:, x_center] == 1)[0].min()
    bottom_edge = np.where(mask[:, x_center] == 1)[0].max()
    # Calculate the centers
    left_center = (y_center, int(alpha * left_edge + antialpha * x_center))
    top_center = (int(alpha * top_edge + antialpha * y_center), x_center)
    bottom_center = (int(alpha * y_center + antialpha * bottom_edge), x_center)

    return left_center, top_center, bottom_center


def draw_roi_mask(mask, radius, *args):
    # Create the mask
    roi_mask = np.zeros_like(mask)
    # Draw the disks
    for center in args:
        rr, cc = disk(center, radius)
        roi_mask[rr, cc] = 1
    roi_mask[mask == 0] = 0
    return roi_mask

def measure_hounsfields_in_mask(image: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """
    A function to measure the average hounsfield value of a segmentation\n
    :param image: the CT image
    :param mask: the segmentation
    :return: the average hounsfield value of the segmentation
    """
    # Get the average hounsfield value of the segmentation
    hounsfield = image[mask == 1]
    return hounsfield.mean(), hounsfield.std()


def measure_liver_hu(series: DicomSeries, model_weights, output_directory, roi_area=5, alpha=0.55) -> Dict:
    """
    measure fat in the liver from a ct scan
    :param series: the dicom series to measure from
    :param model_weights: the path to the model weights used to segment the liver
    :param output_directory: the directory to save the liver roi images to
    :param roi_area: the area of the ROIs to make in cm^2 (default: 5)
    :param alpha: the percent distance from the center to the edge to measure the ROI (default: 0.55)
    :return: a dictionary containing the quantified measures
    """
    filename = f'{series.mrn}_{series.accession}_{series.cut}'
    # Compile the model and load liver segmentation weights
    model = build_unet((512, 512, 1), base_filter=32)
    model.load_weights(model_weights)
    # Preprocess the image
    original_series = series.pixel_array.copy()
    model_image = normalize_image(original_series)
    model_image = np.expand_dims(model_image, axis=-1)
    # Segment the liver
    liver_seg = model.predict(model_image)
    K.clear_session()
    liver_seg = OrganSeg(liver_seg, 'liver', series.spacing, roi_area=roi_area)
    # create lists to hold the appropriate values
    names = ['superior', 'center', 'inferior']
    indices = [liver_seg.superior_point, liver_seg.center_point, liver_seg.inferior_point]
    masks = [liver_seg.superior_slice, liver_seg.center_slice, liver_seg.inferior_slice]
    # measure hounsfield units of various measurments
    hounsfield_data = {}
    vol_mean, _ = measure_hounsfields_in_mask(original_series, liver_seg.seg)
    hounsfield_data['liver_volume_hu_mean'] = vol_mean
    # Measure slicewise hounsfield units
    for idx, mask, name in zip(indices, masks, names):
        ct_image = original_series[idx[0]].copy()
        mean, _ = measure_hounsfields_in_mask(ct_image, mask)
        hounsfield_data[f'{name}_slice_hu_mean'] = mean
        # hounsfield_data[f'{name}_slice_hu_std'] = std
        # Measure the hounsfield units of the ROIs
        numba_idx = List()
        [numba_idx.append(i) for i in idx]
        left, top, bottom = find_roi_centers(mask, numba_idx, alpha=alpha)
        roi_mask = draw_roi_mask(mask, liver_seg.pixel_radius, left, top, bottom)
        mean, std = measure_hounsfields_in_mask(ct_image, roi_mask)
        hounsfield_data[f'{name}_slice_roi_hu_mean'] = mean
        # hounsfield_data[f'{name}_slice_roi_hu_std'] = std
        check = ct_image.copy()
        check[roi_mask == 1] = 255
        fig = plt.imshow(check, cmap='gray')
        plt.savefig(f'{output_directory}/{filename}_{name}_roi.png')
    return hounsfield_data


def measure_spleen_hu(series: DicomSeries, model_weights, roi_area=5, alpha=0.55) -> Dict:
    """
    measure fat in the spleen from a ct scan
    :param series: the dicom series to measure from
    :param model_weights: the path to the model weights used to segment the spleen
    :param roi_area: the area of the ROIs to make in cm^2 (default: 5)
    :param alpha: the percent distance from the center to the edge to measure the ROI (default: 0.55)
    :return: a dictionary containing the quantified measures
    """
    # Compile the model and load liver segmentation weights
    model = build_unet((512, 512, 1), base_filter=32)
    model.load_weights(model_weights)
    # Preprocess the image
    original_series = series.pixel_array.copy()
    model_image = normalize_image(original_series)
    model_image = np.expand_dims(model_image, axis=-1)
    # Segment the spleen
    spleen_seg = model.predict(model_image)
    K.clear_session()
    spleen_seg = OrganSeg(spleen_seg, 'spleen', series.spacing, roi_area=roi_area)
    # create lists to hold the appropriate values
    names = ['superior', 'center', 'inferior']
    indices = [spleen_seg.superior_point, spleen_seg.center_point, spleen_seg.inferior_point]
    masks = [spleen_seg.superior_slice, spleen_seg.center_slice, spleen_seg.inferior_slice]
    # measure hounsfield units of various measurments
    hounsfield_data = {}
    vol_mean, _ = measure_hounsfields_in_mask(original_series, spleen_seg.seg)
    hounsfield_data['spleen_volume_hu_mean'] = vol_mean
    # Measure slicewise hounsfield units
    for idx, mask, name in zip(indices, masks, names):
        ct_image = original_series[idx[0]].copy()
        mean, _ = measure_hounsfields_in_mask(ct_image, mask)
        hounsfield_data[f'spleen_{name}_slice_hu_mean'] = mean
        # hounsfield_data[f'{name}_slice_hu_std'] = std
        # Measure the hounsfield units of the ROIs
        center = tuple(idx[2], idx[1])
        roi_mask = draw_roi_mask(mask, spleen_seg.pixel_radius, center)
        mean, _ = measure_hounsfields_in_mask(ct_image, roi_mask)
        hounsfield_data[f'spleen_{name}_slice_roi_hu_mean'] = mean
    return hounsfield_data
