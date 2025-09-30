"""Unified contour hierarchy extraction and polygon assembly.

Workflow:
1. extract_contour_tree(mask) -> ContourTree (contours + hierarchy + depth + children)
2. assemble_polygons(tree, build_even=True)  -> land-view polygons (even-depth shells, odd holes)
   assemble_polygons(tree, build_even=False) -> inland water polygons (odd-depth shells, even holes)

Parity model (mask is land=1):
- Even depth: land shells (continents, islands in lakes-in-islands, etc.)
- Odd depth: water holes (lakes, inland seas, moats)

Only direct opposite‑parity children become holes; deeper alternations emerge as separate shells.
Invalid rings are repaired via buffer(0). Min filters applied by points and area.
=========================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

__all__ = [
    "ContourTree",
    "extract_contour_tree",
    "assemble_polygons",
]

@dataclass
class ContourTree:
    contours: list
    hierarchy: Optional[np.ndarray]
    depth: list
    children: list

def extract_contour_tree(mask: np.ndarray) -> ContourTree:
    """Extract full contour tree (including hierarchy) from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0/1, 0/255, or any non-zero = foreground).

    Returns
    -------
    ContourTree
        Flat structure containing contours, hierarchy, per-contour depth, and child lists.
    """
    bin_img = (mask > 0).astype("uint8") * 255
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return ContourTree([], None, [], [])
    H = hierarchy[0]  # shape (N,4)
    N = len(contours)
    depth = [0] * N
    for i in range(N):
        d = 0
        p = H[i, 3]
        while p != -1:
            d += 1
            p = H[p, 3]
        depth[i] = d
    children = [[] for _ in range(N)]
    for i in range(N):
        c = H[i, 2]
        while c != -1:
            children[i].append(c)
            c = H[c, 0]
    return ContourTree(contours, hierarchy, depth, children)

def _cnt_coords(cnt):
    return [(int(x), int(y)) for x, y in cnt[:, 0, :]]

def assemble_polygons(tree: ContourTree, *, build_even=True, min_area=5.0, min_points=4, verbose=False):
    """Assemble polygons (with first-level holes) from a ContourTree.

    Parameters
    ----------
    tree : ContourTree
        The contour tree.
    build_even : bool
        If True, build polygons from even-depth contours (land view). If False, odd-depth (water view).
    min_area : float
        Minimum polygon area.
    min_points : int
        Minimum vertex count in a ring.
    verbose : bool
        Print diagnostics for skipped geometries.
    """
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity

    contours = tree.contours
    depth = tree.depth
    children = tree.children
    polys = []
    for i, cnt in enumerate(contours):
        if ((depth[i] % 2) == 0) != build_even:
            continue
        if len(cnt) < min_points:
            continue
        shell = _cnt_coords(cnt)
        holes = []
        for ch in children[i]:
            # hole parity is opposite of shell parity
            if ((depth[ch] % 2) == 0) == build_even:
                continue
            if len(contours[ch]) >= min_points:
                holes.append(_cnt_coords(contours[ch]))
        poly = Polygon(shell, holes)
        if (not poly.is_valid) or (poly.area < min_area):
            fixed = poly.buffer(0)
            if fixed.is_empty:
                continue
            if fixed.geom_type == "Polygon":
                cands = [fixed]
            else:
                cands = [g for g in fixed.geoms if g.area >= min_area]
        else:
            cands = [poly]
        for g in cands:
            if g.is_valid and g.area >= min_area:
                polys.append(g)
            elif verbose:
                print("Skipped invalid piece:", explain_validity(g))
    return polys
