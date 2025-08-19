import cv2
import numpy as np
from typing import List, Tuple

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

def _depth_and_children(hierarchy) -> Tuple[List[int], List[List[int]]]:
    """
    Extract depth and children information from the contour hierarchy.
    """
    print(f"{hierarchy=}")

    H = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]
    N = H.shape[0]
    depth = [-1]*N
    children = [[] for _ in range(N)]
    for i in range(N):
        p = H[i][3]; d = 0
        while p != -1:
            d += 1
            p = H[p][3]
        depth[i] = d
        c = H[i][2]
        while c != -1:
            children[i].append(c)
            c = H[c][0]

    return depth, children
