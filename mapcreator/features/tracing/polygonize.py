"""Unified polygonization helpers.

This module now delegates to the unified contour extraction + polygon
assembly logic in ``contour_extraction``. The previous separate functions
``contours_to_polygons`` and ``contours_to_polygons_with_holes`` are retained
as thin wrappers for compatibility but internally reuse shared code.
"""

from typing import Optional
from mapcreator.globals.configs import MIN_POINTS
from .contour_extraction import ContourTree, assemble_polygons

# New unified public helpers (optional explicitness)
def land_polygons_from_tree(
    tree: ContourTree,
    meta: Optional[dict] = None,
    *,
    min_area: float | None = None,
    min_points: int | None = None,
    verbose: bool = False,
):
    """Return land (even-depth) polygons, pulling defaults from meta if provided."""
    meta = meta or {}
    if min_area is None:
        min_area = meta.get("min_area", 5.0)
    if min_points is None:
        min_points = meta.get("min_points", MIN_POINTS)
    return assemble_polygons(tree, build_even=True, min_area=min_area, min_points=min_points, verbose=verbose)

def water_polygons_from_tree(
    tree: ContourTree,
    meta: Optional[dict] = None,
    *,
    min_area: float | None = None,
    min_points: int | None = None,
    verbose: bool = False,
):
    """Return inland water (odd-depth) polygons, pulling defaults from meta if provided."""
    meta = meta or {}
    if min_area is None:
        min_area = meta.get("min_area", 5.0)
    if min_points is None:
        min_points = meta.get("min_points", MIN_POINTS)
    return assemble_polygons(tree, build_even=False, min_area=min_area, min_points=min_points, verbose=verbose)