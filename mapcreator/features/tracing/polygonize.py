"""
mapcreator/features/tracing/polygonize.py

Unified polygonization helpers.

This module now delegates to the unified contour extraction + polygon
assembly logic in ``contour_extraction``. The previous separate functions
``contours_to_polygons`` and ``contours_to_polygons_with_holes`` are retained
as thin wrappers for compatibility but internally reuse shared code.
"""

from typing import Optional


from mapcreator.globals.configs import MIN_POINTS, MIN_AREA
from mapcreator.features.tracing.contour_extraction import ContourTree, _cnt_coords

from shapely.geometry import Polygon
from shapely.validation import explain_validity

def assemble_polygons(tree: ContourTree, 
                      *, build_even=True, min_area=MIN_AREA,  min_points=MIN_POINTS, 
                      verbose=False,) -> list[Polygon]:
    
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

    Returns
    -------
    list[Polygon]
        Valid Shapely polygons meeting filters. Invalid rings are repaired via buffer(0).
    """

    contours = tree.contours
    depth = tree.depth
    children = tree.children
    
    if verbose:
        print(f"assemble_polygons(build_even={build_even}, min_area={min_area}, min_points={min_points})")

    polys: list[Polygon] = []

    for i, cnt in enumerate(contours):
        if ((depth[i] % 2) == 0) != build_even:
            if verbose:
                print(f"Skipped contour {i} with depth {depth[i]} (build_even={build_even})")
            continue
        if len(cnt) < min_points:
            if verbose:
                print(f"Skipped contour {i} with {len(cnt)} points (<{min_points})")
            continue

        shell = _cnt_coords(cnt)
        holes: list[list[tuple[int, int]]] = []

        # Only direct opposite-parity children become holes
        for ch in children[i]:
            # hole parity is opposite of shell parity
            if ((depth[ch] % 2) == 0) == build_even:
                if verbose:
                    print(f"Skipped child contour {ch} with depth {depth[ch]} (same parity as shell)")
                continue
            if len(contours[ch]) >= min_points:
                if verbose:
                    print(f"Added hole contour {ch} with {len(contours[ch])} points")
                holes.append(_cnt_coords(contours[ch]))

        poly = Polygon(shell, holes)

        # Repair invalid rings and split multiparts, then filter
        if (not poly.is_valid) or (poly.area < min_area):
            fixed = poly.buffer(0)
            if fixed.is_empty:
                continue
            candidates = (
                [fixed] if fixed.geom_type == "Polygon"
                else [g for g in fixed.geoms if g.area >= min_area]
            )
        else:
            candidates = [poly]
        
        for g in candidates:
            if g.is_valid and g.area >= min_area:
                polys.append(g)
            elif verbose:
                print("Skipped invalid piece:", explain_validity(g))

    return polys

def build_polygons_from_tree(
    tree: ContourTree,
    meta: Optional[dict] = None,
    *,
    min_area: float | None = None,
    min_points: int | None = None,
    verbose: bool = False,
) -> tuple[list[Polygon], list[Polygon]]:
    """Return all polygons (land + inland water), pulling defaults from meta if provided.
       Wrapper around assemble_polygons() to get both even and odd parity contour sets.
    Parameters
    ----------
    tree : ContourTree
        The contour tree.
    meta : dict, optional
        Optional metadata dictionary to pull default min_area and min_points from.
    min_area : float, optional
        Minimum polygon area. If None, pulls from meta or defaults to MIN_AREA.
    min_points : int, optional
        Minimum vertex count in a ring. If None, pulls from meta or defaults to MIN_POINTS.
    verbose : bool
        Print diagnostics for skipped geometries.
    """
    meta = meta or {}
    if min_area is None:
        min_area = meta.get("min_area", MIN_AREA)
    if min_points is None:
        min_points = meta.get("min_points", MIN_POINTS)

    # Even-depth shells = land polygons
    land_polys = assemble_polygons(
        tree, build_even=True, min_area=min_area, min_points=min_points, verbose=verbose
    )

    # Odd-depth shells = inland water polygons
    water_polys = assemble_polygons(
        tree, build_even=False, min_area=min_area, min_points=min_points, verbose=verbose
    )

    return land_polys, water_polys

if __name__ == "__main__":
    """Simple test and visualization"""
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from mapcreator.features.tracing.contour_extraction import extract_contour_tree, _plot_contour_tree, contour_tree_diagnostics

    # Example usage and test
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (180, 180), 1, -1)  # Large square
    cv2.circle(img, (30, 30), 10, 0, -1)              # Hole touching
    cv2.circle(img, (60, 60), 30, 0, -1)              # Hole in square
    cv2.circle(img, (140, 60), 20, 0, -1)             # Another hole
    cv2.circle(img, (100, 125), 40, 0, -1)            # Another hole
    cv2.circle(img, (100, 140), 10, 1, -1)            # Island in hole

    tree = extract_contour_tree(img)
    contour_tree_diagnostics(tree)

    _plot_contour_tree(tree)
    plt.show()

    land_polys = assemble_polygons(tree, build_even=True, verbose=True)
    water_polys = assemble_polygons(tree, build_even=False, verbose=True)
    print(f"Extracted {len(land_polys)} land polygons and {len(water_polys)} water polygons.")
    print('-'*100)