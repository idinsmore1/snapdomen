import numpy as np
import keras.backend as K
from scipy.ndimage import binary_fill_holes, binary_erosion
import matplotlib.pyplot as plt
from .model import build_unet
from snapdomen.imaging.utils import normalize_image
from snapdomen.waistcirc.measure import get_largest_connected_component, get_waist_circumference, remove_artifacts
from snapdomen.imaging.dicomseries import DicomSeries


def extract_abdomen(pixel_array, start_point, end_point):
    """
    Only extract the slices that are included in the abdomen
    :param pixel_array: the pixel data that make up the ct scan
    :param start_point: start point axial index (usually the l1 vertebra)
    :param end_point: end point axial index (usually the l5 vertebra)
    :return: the extracted abdomen array
    """
    abdomen = pixel_array[start_point: end_point, :, :].copy()
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


def predict_abdomen(series: DicomSeries, start: int, end: int, model_weights: str):
    """
    Predict the abdomen segmentation from a ct scan
    :param series: the ct scan
    :param start: the start axial index
    :param end: the end axial index
    :param model_weights: the path to the model weights
    :return: the abdomen segmentation
    """
    abdomen = extract_abdomen(series.pixel_array, start, end)
    abdomen_norm = normalize_image(abdomen)[..., np.newaxis]
    model = build_unet((512, 512, 1), base_filter=32)
    model.load_weights(model_weights)
    preds = model.predict(abdomen_norm)
    K.clear_session()
    preds = postprocess_prediciton(preds)
    return preds


def save_abdominal_wall_overlay(image, mask, output_path):
    """
    Save the abdominal wall overlay
    :param image: the original axial image slice from the ct scan
    :param mask: the abdominal wall mask
    :param output_path: the path to save the overlay
    :return: None
    """
    plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray', interpolation='none')
    # plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.imshow(mask, alpha=0.5, cmap='jet', interpolation='none')
    plt.savefig(output_path)
    plt.close()


def quantify_abdominal_fat(series, start, end, l3, model_weights, outdir):
    """
    Quantify the fat in the abdomen from a ct scan
    :param series: the ct scan
    :param start: the start axial index
    :param end: the end axial index
    :param l3: the l3 vertebra axial index
    :param model_weights: the path to the model weights
    :param outdir: the output directory
    :return: the fat measurements for each slice
    """
    preds = predict_abdomen(series, start, end, model_weights)
    measurements = {}
    for i in range(start, end):
        image = series.pixel_array[i].copy()
        pred = preds[i - start]
        image = remove_artifacts(image)
        interior, exterior = separate_abdominal_cavity(image, pred)
        visceral_fat_pixels, visceral_fat_area = measure_fat(image, interior, series.spacing[0], series.spacing[1])
        subcutaneous_fat_pixels, subcutaneous_fat_area = measure_fat(image, exterior, series.spacing[0],
                                                                     series.spacing[1])
        save_im = [True if i in [start, end - 1, l3] else False][0]
        if save_im:
            save_abdominal_wall_overlay(image, interior, f"{outdir}/MRN{series.mrn}_{series.accession}_{series.cut}_slice_{i}_abdominal_wall.png")
        _, wc = get_waist_circumference(series, i, save_im=save_im, outdir=outdir)
        measurements[f'slice_{i}'] = {
            'waist_circumference': float(wc),
            'visceral_fat_pixels': int(visceral_fat_pixels),
            'visceral_fat_area': float(visceral_fat_area),
            'subcutaneous_fat_pixels': int(subcutaneous_fat_pixels),
            'subcutaneous_fat_area': float(subcutaneous_fat_area)
        }
    return measurements
