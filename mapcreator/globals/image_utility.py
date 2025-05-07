from mapcreator.globals import configs
from mapcreator.globals import directories
import numpy as np
import cv2

def flood_fill_img(img_array: np.ndarray, seedPoint=(0, 0)) -> np.ndarray:
    """
    Flood fill from top left hand pixel
    Removes the ocean from a binary water image by flood-filling from the edge.

    Returns a new array where only inland lakes/seas are white (255),
    and ocean (connected to edge) is black (0).
    """
    filled = img_array.copy()
    h, w = filled.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Fill from top-left (or any edge pixel known to be ocean)
    cv2.floodFill(filled, mask, seedPoint=seedPoint, newVal=0)

    return filled
