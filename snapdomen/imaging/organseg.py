import numpy as np
from numba import njit
from scipy.ndimage import binary_erosion
from itertools import groupby
from operator import itemgetter
from typing import List as ListType
from typing import Tuple


class OrganSeg:
    """
    A class representing the segmentation of an organ in a CT scan\n
    """
    # Distance between slices in mm to measure
    DISTANCE_BETWEEN_SLICES = 20

    def __init__(self, segmentation: np.ndarray, organ: str, spacing: ListType[float], roi_area: float = 2.5):
        """
        :param segmentation: the output from the UNet
        :param organ: the name of the organ being segmented
        :param spacing: the pixel spacing of the CT scan
        :param roi_area: the desired area of the ROI in cm^2 (default: 2.5)
        """
        # self.original_seg = segmentation
        self.max_slice = segmentation.shape[0] - 1
        self.min_slice = 0
        self.organ = organ
        # Set the threshold for
        if organ == 'spleen':
            self.threshold = 100
        else:
            self.threshold = 0

        if organ == 'spleen':
            self.n_rois = 1
        else:
            self.n_rois = 3

        self.pixel_radius = self.calculate_pixel_radius(spacing[0], roi_area)
        self.seg = self._prepare_seg_output(segmentation, self.threshold)
        self.center_point = self._erode_seg(self.seg)
        self.center_slice = self.seg[self.center_point[0], :, :].copy()

        # get the superior and inferior slices
        # measure distance is the number of slices above and below center slice to measure
        self.measure_distance = OrganSeg.DISTANCE_BETWEEN_SLICES // spacing[-1]
        if self.center_point[0] - self.measure_distance < self.min_slice:
            self.superior_slice = self.seg[self.min_slice, :, :].copy()
            self.superior_point = [self.min_slice, self.center_point[1], self.center_point[2]]
        else:
            self.superior_slice = self.seg[int(self.center_point[0] - self.measure_distance), :, :].copy()
            self.superior_point = [int(self.center_point[0] - self.measure_distance), self.center_point[1],
                                   self.center_point[2]]

        if self.center_point[0] + self.measure_distance > self.max_slice:
            self.inferior_slice = self.seg[self.max_slice, :, :].copy()
            self.inferior_point = [self.max_slice, self.center_point[1], self.center_point[2]]
        else:
            self.inferior_slice = self.seg[int(self.center_point[0] + self.measure_distance), :, :].copy()
            self.inferior_point = [int(self.center_point[0] + self.measure_distance), self.center_point[1],
                                   self.center_point[2]]

    @staticmethod
    def _longest_consecutive_seg(numbers: ListType):
        """
        A function to find the longest consecutive cut of liver from a segmentation\n
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

    def _prepare_seg_output(self, segmentation: np.ndarray, threshold: int = 0):
        """
        Postprocess the organ segmentation from the UNet\n
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
        start, end = self._longest_consecutive_seg(has_organ)
        im_start = has_organ[start]
        im_end = has_organ[end+1]
        # Create the final segmentation
        # Set any slices that don't have liver to zero
        seg[:im_start, :, :] = 0
        seg[im_end:, :, :] = 0
        return seg

    @staticmethod
    def _erode3d(seg: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _erode_seg(seg_erode: np.ndarray, erosion_limit=100) -> np.ndarray:
        """
        A function to erode down a segmentation using erode3d function\n
        :param seg_erode: segmentation of original size to erode
        :param erosion_limit: limit to stop eroding the image down (default: 100)
        :return: a numpy array containing the center point
        """
        i = 0
        erode_last = seg_erode
        while np.sum(seg_erode) > erosion_limit:
            erode_last = seg_erode
            seg_erode = OrganSeg._erode3d(seg_erode)
            i += 1

        if seg_erode.sum() == 0:
            seg_erode = erode_last

        single = np.argwhere(seg_erode)
        single = single[int(len(single) / 2)]
        return single

    @staticmethod
    @njit
    def calculate_pixel_radius(width: float, roi_area):
        """
        Calculate the radius of an ROI in pixels
        :param width: the width of the pixel in mm
        :param roi_area: desired area of ROI in centimeters
        :return: radius in pixels for the ROI
        """
        # Get the pixel width in centimeters
        pixel_width = width / 10
        radius = np.sqrt(roi_area / np.pi)
        pixel_radius = int(np.ceil(radius / pixel_width))
        return pixel_radius
