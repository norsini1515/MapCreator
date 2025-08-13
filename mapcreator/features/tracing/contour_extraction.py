import cv2
import numpy as np


def find_external_contours(bin_img: np.ndarray):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_tree_contours(bin_img: np.ndarray):
    """Return (contours, hierarchy) with full nesting info."""
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def contours_from_binary_mask(mask: np.ndarray):
    mask = (mask > 0).astype("uint8") * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def contours_from_binary_mask_tree(mask: np.ndarray):
    mask = (mask > 0).astype("uint8") * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy