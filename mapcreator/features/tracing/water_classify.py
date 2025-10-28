"""
mapcreator/features/tracing/water_classify.py

Water classification and inland/ocean split.
This module provides functions to distinguish ocean from inland water
using flood-fill, and to polygonize inland water with nesting (islands in lakes).
"""
import numpy as np
import cv2
from .contour_extraction import contours_from_binary_mask_tree
from .polygonize import contours_to_polygons_with_holes


def split_ocean_vs_inland(water_mask: np.ndarray):
    """
    water_mask: 0/1 ndarray where 1 = water
    Returns: (ocean_mask, inland_mask) as 0/1 arrays using border-connected flood-fill.
    """
    h, w = water_mask.shape
    padded = np.pad(water_mask.astype("uint8"), 1, mode="constant", constant_values=0)
    ff = (padded * 255).astype("uint8")
    H, W = ff.shape
    # flood fill from borders marking ocean as 128
    for x in range(W):
        cv2.floodFill(ff, None, (x, 0), 128)
        cv2.floodFill(ff, None, (x, H - 1), 128)
    for y in range(H):
        cv2.floodFill(ff, None, (0, y), 128)
        cv2.floodFill(ff, None, (W - 1, y), 128)
    ff = ff[1:-1, 1:-1]
    ocean_mask = (ff == 128).astype("uint8")
    inland_mask = ((water_mask == 1) & (ocean_mask == 0)).astype("uint8")
    return ocean_mask, inland_mask

def inland_mask_to_polygons(inland_mask: np.ndarray, *, min_area=5.0, min_points=3):
    """Polygonize inland water with nesting (islands inside lakes)."""
    contours, hierarchy = contours_from_binary_mask_tree(inland_mask)
    return contours_to_polygons_with_holes(contours, hierarchy, min_area=min_area, min_points=min_points)

