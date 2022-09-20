import numpy as np
from numba import njit

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

