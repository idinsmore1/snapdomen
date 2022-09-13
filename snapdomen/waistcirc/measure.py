import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, binary_erosion
from skimage.measure import label, find_contours, regionprops
from typing import Tuple, Any


# @njit
def binarize_image(image: np.ndarray, threshold: int = -500) -> np.ndarray:
    """
    A function to binarize a ct scan slice between air and body \n
    :param image: dicom pixel image
    :param threshold: the threshold value to binarize the image (default: -500)
    :return: the binary image as a numpy array
    """
    binary_image = (image.copy() > threshold) * 1
    binary_image = binary_fill_holes(binary_image)
    return binary_image


def get_largest_connected_component(binary_image: np.ndarray) -> np.ndarray:
    """
    A function to get the largest connected component from binary image (in this case it should be the body)\n
    :param binary_image: the output from binarize_image
    :return: a numpy array containing only the largest connected component
    """
    labels = label(binary_image)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]  # the 0 label is by default background so take the rest
    largest = max(list_seg, key=lambda x: x[1])[0]
    labels_max = (labels == largest).astype(int)
    return labels_max


def remove_exterior_artifacts(ct_image: np.ndarray, largest_connected_component: np.ndarray) -> np.ndarray:
    """
    A function to remove artifacts outside the body from CT scans\n
    :param ct_image: the original ct image
    :param largest_connected_component: the output from get_largest_connected_component
    :return: the cleaned up original image
    """
    ct_image[largest_connected_component != 1] = np.min(ct_image.copy())
    return ct_image


def measure_circumference(body_array: np.ndarray, pixel_width: float, pixel_units: str = 'mm'):
    """
    Measure the circumference of the body in centimeters\n
    :param body_array: the output from binarize_image
    :param pixel_width: the width of a pixel in centimeters
    :param pixel_units: the units of the pixel width (default: mm)
    :return: the circumference of the body in centimeters
    """
    labels = label(body_array)
    regions = regionprops(labels)
    waist_perimeter = max(regions, key=lambda x: x.area).perimeter
    # waist_pixels = perimeter(body_array, neighbourhood=4)
    waist_cm = np.round(waist_perimeter * pixel_width, 2)
    if pixel_units == 'mm':
        waist_cm = waist_cm / 10
    return np.round(waist_cm)


def get_waist_circumference(axial_array: np.ndarray, l3_slice: np.ndarray, spacing: list) -> Tuple[Any, Any]:
    """
    A function to measure the circumference of the body in centimeters\n
    :param axial_array: the axial ct scan series
    :param l3_slice: the index of the l3 axial slice
    :param spacing: the spacing of the ct scan in millimeters
    :return: the circumference of the body in centimeters
    """
    l3_image = axial_array[l3_slice]
    # Remove exterior artifacts
    binary_l3 = binarize_image(l3_image)
    body = get_largest_connected_component(binary_l3)
    l3_pp = remove_exterior_artifacts(l3_image, body)
    # Measure circumference
    l3_pp = binarize_image(l3_pp)
    # Erosion is to remove small artifacts attached to body (clips from table, etc
    l3_pp = binary_erosion(l3_pp, iterations=4)
    waist_circ = measure_circumference(l3_pp, spacing[0])
    return l3_pp, waist_circ


def plot_contours(original_image, body_mask, output_path):
    """
    A function to plot the contours of the waist circumference around original image
    :param original_image: the original L3 slice
    :param body_mask: the body mask
    :param output_path: the path to save the plot
    :return: the plot of the contours
    """
    contours = find_contours(body_mask)
    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=3)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path)
