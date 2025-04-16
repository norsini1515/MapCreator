# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 00:08:16 2025

@author: nichorsin598
"""
import cv2
import numpy as np
from pathlib import Path

def flood_fill_ocean(img_array: np.ndarray) -> np.ndarray:
    """
    Removes the ocean from a binary water image by flood-filling from the edge.

    Returns a new array where only inland lakes/seas are white (255),
    and ocean (connected to edge) is black (0).
    """
    filled = img_array.copy()
    h, w = filled.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Fill from top-left (or any edge pixel known to be ocean)
    cv2.floodFill(filled, mask, seedPoint=(0, 0), newVal=0)

    return filled