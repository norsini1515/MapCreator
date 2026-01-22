"""
mapcreator/features/tracing/contour_extraction.py

Unified contour hierarchy extraction and polygon assembly.

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
from typing import List, Tuple, Optional, Any, cast

__all__ = [
    "ContourTree",
    "extract_contour_tree",
    "contour_tree_diagnostics"
]

# Alias for clarity. Each OpenCV contour is (N, 1, 2) int array of xy pairs.
Contour = np.ndarray

@dataclass
class ContourTree:
    """
    Flat structure holding the contour hierarchy and simple navigational aids.

    Attributes
    ----------
    contours : list[Contour]
        Each contour is an (N,1,2) ndarray of integer coordinates.
    hierarchy : Optional[np.ndarray]
        Shape (N, 4). Columns are [next, prev, first_child, parent] per OpenCV.
        None when no contours are found.
    depth : list[int]
        Nesting depth for each contour. Roots are depth 0. Depth increments by 1
        for each parent step.
    children : list[list[int]]
        Direct children indices for each contour, following the "first_child" and
        sibling chain via "next".
    """
    contours: list
    hierarchy: Optional[np.ndarray]
    depth: list
    children: list

def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize an input mask into uint8 {0,255} where nonzero becomes 255.
    Accepts 0/1, 0/255, or arbitrary nonzero inputs.
    """
    if mask.ndim != 2:
        # If caller passes a 3-channel mask, reduce it safely.
        mask = mask[..., 0]
    return np.where(mask > 0, 255, 0).astype("uint8")

def _compute_depths(hier: np.ndarray) -> List[int]:
    """
    Compute depth for each contour by walking parent pointers.
    hier: shape (N,4) with columns [next, prev, first_child, parent]
    """
    N = hier.shape[0]
    depth = [0] * N
    for i in range(N):
        d = 0
        p = hier[i, 3]  # parent
        while p != -1:
            d += 1
            p = hier[p, 3]
        depth[i] = d
    return depth

def _compute_children(hier: np.ndarray) -> List[List[int]]:
    """
    Build per-node direct children lists using first_child + next sibling chain.
    """
    N = hier.shape[0]
    children: List[List[int]] = [[] for _ in range(N)]
    for i in range(N):
        child = hier[i, 2]  # first_child
        while child != -1:
            children[i].append(child)
            child = hier[child, 0]  # next sibling
    return children

def extract_contour_tree(mask: np.ndarray, verbose: bool = False) -> ContourTree:
    """
    Extract the full contour tree from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary-ish mask. Any nonzero is treated as foreground.
    verbose : bool
        If True, print diagnostics.
    Returns
    -------
    ContourTree
        Contours, OpenCV hierarchy, per-contour depth, and direct children lists.
    """
    bin_img = _ensure_binary_mask(mask)
    # OpenCV 3: returns (image, contours, hierarchy)
    # OpenCV 4: returns (contours, hierarchy)
    fc_result = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if isinstance(fc_result, tuple) and len(fc_result) == 3:
        _, contours, hierarchy = cast(Tuple[Any, List[Contour], Optional[np.ndarray]], fc_result)
    else:
        contours, hierarchy = cast(Tuple[List[Contour], Optional[np.ndarray]], fc_result)

    # No contours found
    if hierarchy is None or len(contours) == 0:
        return ContourTree(contours=[], hierarchy=None, depth=[], children=[])

    hier = hierarchy[0]  # shape (N,4) [Next, Previous, First_Child, Parent]

    depths = _compute_depths(hier)
    kids = _compute_children(hier)

    tree = ContourTree(contours, hier, depths, kids)
    
    if verbose:
        _plot_contour_tree(tree)
        # contour_tree_diagnostics(tree)

    return tree

def _cnt_coords(cnt: Contour) -> List[Tuple[int, int]]:
    """Convenience: return a simple list of (x, y) from an OpenCV contour."""
    # cnt has shape (N,1,2)
    return [(int(x), int(y)) for x, y in cnt[:, 0, :]]

def _plot_contour_tree(tree: ContourTree, ax=None):
    """Visualize the contour tree over a blank canvas, color-coded by depth."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    if ax is None:
        fig, ax = plt.subplots()

    max_depth = max(tree.depth) if tree.depth else 0
    norm = Normalize(vmin=0, vmax=max_depth)
    cmap = cm.get_cmap("viridis", max_depth + 1)

    for i, cnt in enumerate(tree.contours):
        d = tree.depth[i]
        color = cmap(norm(d))
        pts = cnt[:, 0, :]
        ax.plot(pts[:, 0], pts[:, 1], color=color, label=f"Depth {d}")

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Contour Tree Visualization")
    if max_depth > 0:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Depth")
    plt.show()

def contour_tree_diagnostics(tree: ContourTree):
    # from matplotlib.patches import Polygon
    from shapely.geometry import Polygon
    
    print("Contour Tree Diagnostics:")
    print(f"Registered {len(tree.contours)} contours from test image.")
    print("Contour details:")
    print(" idx  depth  npts     area       role")
    for i, cnt in enumerate(tree.contours):
        n = len(cnt)
        d = tree.depth[i]
        shell = _cnt_coords(cnt)
        area = abs(Polygon(shell).area)
        print(f"idx={i:2d} depth={d} npts={n:4d} area={area:8.1f} role={'water-shell' if d%2 else 'land-shell'}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Simple test case
    test_mask = np.zeros((200, 200), dtype="uint8")
    cv2.rectangle(test_mask, (20, 20), (180, 180), 1, -1)  # big square
    cv2.circle(test_mask, (60, 60), 30, 0, -1)             # hole
    cv2.circle(test_mask, (140, 60), 20, 0, -1)            # hole
    cv2.circle(test_mask, (100, 145), 40, 0, -1)           # hole
    cv2.circle(test_mask, (100, 140), 15, 1, -1)           # island in hole

    tree = extract_contour_tree(test_mask)

    contour_tree_diagnostics(tree)

    # Visualize
    plt.imshow(test_mask, cmap="gray")
    for i, cnt in enumerate(tree.contours):
        pts = cnt[:, 0, :]
        plt.plot(pts[:, 0], pts[:, 1], label=f"Depth {tree.depth[i]}")
    plt.legend()
    plt.show()