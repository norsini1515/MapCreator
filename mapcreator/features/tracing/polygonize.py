"""
mapcreator/features/tracing/polygonize.py

Unified polygonization helpers.

This module now delegates to the unified contour extraction + polygon
assembly logic in ``contour_extraction``. The previous separate functions
``contours_to_polygons`` and ``contours_to_polygons_with_holes`` are retained
as thin wrappers for compatibility but internally reuse shared code.
"""

from typing import Optional
from affine import Affine

from mapcreator.globals.configs import MIN_POINTS, MIN_AREA
from mapcreator.features.tracing.contour_extraction import ContourTree, _cnt_coords
from mapcreator.globals.logutil import info, process_step, error, warn, setting_config
from mapcreator.features.tracing.geo_transform import apply_affine_to_gdf

from shapely.ops import unary_union
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity

import geopandas as gpd

def _extent_polygon(meta: dict):
    # Build extent in PIXEL space (0..width, 0..height). We transform to map coords later.
    process_step("Building extent polygon in pixel space")
    setting_config(f"Extent polygon bounds: (0, 0, {meta['width']}, {meta['height']})")

    return box(0, 0, meta["width"], meta["height"])

def _is_shell_of_parity(depth_val: int, build_even: bool) -> bool:
    """True if contour with depth_val should be treated as a shell for this build."""
    return (depth_val % 2 == 0) == build_even

def _collect_holes_for_shell(i: int, tree: ContourTree, *, min_points: int, build_even: bool, verbose: bool):
    """Collect immediate opposite-parity child rings for contour i."""
    holes: list[list[tuple[int, int]]] = []
    contours = tree.contours
    depth = tree.depth
    for ch in tree.children[i]:
        # opposite parity to shell
        if _is_shell_of_parity(depth[ch], build_even):
            continue
        if len(contours[ch]) >= min_points:
            if verbose:
                info(f"Added hole contour {ch} with {len(contours[ch])} points")
            holes.append(_cnt_coords(contours[ch]))
    return holes

def _repair_and_split(poly: Polygon, *, min_area: float, verbose: bool):
    """Fix invalids via buffer(0), split multiparts, yield valid pieces >= min_area."""
    if (not poly.is_valid) or (poly.area < min_area):
        fixed = poly.buffer(0)
        if fixed.is_empty:
            return
        if fixed.geom_type == "Polygon":
            pieces = [fixed]
        else:
            pieces = [g for g in fixed.geoms if g.area >= min_area]
    else:
        pieces = [poly]
    for g in pieces:
        if g.is_valid and g.area >= min_area:
            yield g
        elif verbose:
            # Only compute expensive message when verbose
            try:
                msg = explain_validity(g)
            except Exception:
                msg = "invalid geometry"
            warn(f"Skipped invalid piece: {msg}")

def assemble_polygons(
    tree: ContourTree,
    *,
    build_even: bool = True,
    min_area: float = MIN_AREA,
    min_points: int = MIN_POINTS,
    verbose: bool = False,
    return_meta: bool = True,
) -> list:
    
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
    list
        If return_meta=False (default): list[Polygon]
        If return_meta=True: list[tuple[Polygon, int]] as (geometry, depth) #TODO in the future depth can get wrapped into a ContourMetadata object
        Valid Shapely polygons meet filters. Invalid rings are repaired via buffer(0).
    """

    process_step(f"Assembling {'land' if build_even else 'inland water'} polygons from contours")

    contours = tree.contours
    depth = tree.depth
    children = tree.children
    

    if verbose:
        setting_config(f"assemble_polygons(build_even={build_even}, min_area={min_area}, min_points={min_points})")

    # Note: we keep the return type flexible for backward compatibility.
    polys: list = []

    for i, cnt in enumerate(contours):
        # keep only shells of the selected parity
        if not _is_shell_of_parity(depth[i], build_even):
            continue

        if len(cnt) < min_points:
            #skip too-small contours
            if verbose:
                warn(f"Skipped contour {i} with {len(cnt)} points (<{min_points})")
            continue

        shell = _cnt_coords(cnt)
        holes = _collect_holes_for_shell(i, tree, min_points=min_points, build_even=build_even, verbose=verbose)
        poly = Polygon(shell, holes)

        for g in _repair_and_split(poly, min_area=min_area, verbose=verbose):
            if return_meta:
                polys.append((g, depth[i]))
            else:
                polys.append(g)

    return polys
        
def build_polygons_from_tree(
    tree: ContourTree,
    meta: Optional[dict] = None,
    *,
    min_area: float | None = None,
    min_points: int | None = None,
    verbose: bool = False,
    return_meta: bool = True,
) -> tuple[list, list]:
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
    return_meta : bool
        If True, return polygons as (geometry, depth) tuples.
    Returns
    -------
    tuple[list, list]
        A tuple of (even_polygons, odd_polygons), where:
        - even_polygons: list of land polygons (even-depth shells)
        - odd_polygons: list of inland water polygons (odd-depth shells)
        If return_meta=True, each element is a list of (geometry, depth) tuples.
    """
    process_step("Building polygons from contour tree")

    meta = meta or {}
    if min_area is None:
        min_area = meta.get("min_area", MIN_AREA)
    if min_points is None:
        min_points = meta.get("min_points", MIN_POINTS)

    # Even-depth shells = land polygons
    even_polys = assemble_polygons(
        tree,
        build_even=True,
        min_area=min_area,
        min_points=min_points,
        verbose=verbose,
        return_meta=return_meta,
    )

    # Odd-depth shells = inland water polygons
    odd_polys = assemble_polygons(
        tree,
        build_even=False,
        min_area=min_area,
        min_points=min_points,
        verbose=verbose,
        return_meta=return_meta,
    )

    return even_polys, odd_polys

def compute_extent_polygon(
    even_polys: list,
    odd_polys: list,
    meta: dict,
    verbose: bool = False,
    return_meta: bool = True,
) -> Polygon:
    
    """Compute the extent polygon by subtracting all land and inland water from the image extent box.
    Parameters
    ----------
    even_polys : list[Polygon]
        Land polygons (even-depth shells).
    odd_polys : list[Polygon]
        Inland water polygons (odd-depth shells).
    meta : dict
        Metadata dictionary containing the "extent" key with xmin, ymin, xmax, ymax.
    verbose : bool
        Print diagnostics.
    return_meta : bool
        If True, return a tuple of (geometry, 0) where 0 is the assigned depth for extent/ocean.
    """
    process_step("Computing extent polygon")
    if "extent" not in meta:
        raise ValueError("Meta missing 'extent' key; cannot compute extent polygon.")


    # unwrap (geom, depth) tuples if needed
    if even_polys and isinstance(even_polys[0], tuple):
        even_polys = [p for p, _ in even_polys]
    if odd_polys and isinstance(odd_polys[0], tuple):
        odd_polys = [p for p, _ in odd_polys]

    # Build extent polygon in pixel space (0..width, 0..height)
    extent_poly = _extent_polygon(meta)

    subtract_geoms = []
    # Add all land and inland water polygons to the subtraction list
    if even_polys:
        subtract_geoms.append(unary_union(even_polys))
    if odd_polys:
        subtract_geoms.append(unary_union(odd_polys))
    if verbose:
        info(f"Subtracting {len(subtract_geoms)} geometries from extent box")

    if subtract_geoms:
        subtract_union = unary_union(subtract_geoms)
        extent_poly = extent_poly.difference(subtract_union)
        
        if verbose:
            info(f"Computed extent polygon by subtracting {len(subtract_geoms)} geometries")
    
    if return_meta:
        # Assign depth=-1 for the extent/ocean polygon
        return (extent_poly, -1)
    
    return extent_poly

def extract_polygons_from_binary(bin_img, meta: dict, verbose: bool = False) -> tuple[list, list]:
    """Convenience wrapper: from a binary land mask to even/odd polygons.

    Even-depth shells correspond to the foreground class in the mask (1s),
    odd-depth shells to the background holes (0s). This retains the general
    even/odd semantics; mapping to domain labels (land/water) happens upstream.

    Returns (even_polys, odd_polys). If meta['compute_extent_polygons'] is True,
    appends the extent/ocean polygon as (geometry, 0) to the odd list.
    """
    from .contour_extraction import extract_contour_tree

    tree = extract_contour_tree(bin_img > 0, verbose=verbose)
    if verbose:
        info(f"Extracted contours: {len(tree.contours)}")
        if tree.depth:
            info(f"Contour max tree depth: {max(tree.depth)}")

    if not tree.contours:
        return [], []

    even_polys, odd_polys = build_polygons_from_tree(
        tree, meta=meta, verbose=verbose, return_meta=True
    )

    if meta.get("compute_extent_polygons", True):
        extent_poly, extent_depth = compute_extent_polygon(
            even_polys=even_polys, odd_polys=odd_polys, meta=meta, verbose=verbose, return_meta=True
        )
        if not extent_poly.is_empty:
            odd_polys.append((extent_poly, extent_depth))
            if verbose:
                info("Added extent polygon to odd set (depth=0)")

    return even_polys, odd_polys

def polygons_to_gdf(polygons, crs:str="EPSG:3857", affine_val:Affine=None, verbose: bool|str = False) -> gpd.GeoDataFrame:
    """Convert a list of Shapely polygons (or (polygon, depth) tuples) to a GeoDataFrame."""
    from mapcreator.features.tracing.gdf_tools import to_gdf

    # Support both tuples and plain geometries
    if isinstance(polygons[0], Polygon):
        if verbose == 'debug':
            setting_config("polygons_to_vector: Detected list of Polygon geometries.") 
        geoms = list(polygons)
        depths = [None] * len(geoms)
    elif isinstance(polygons[0], tuple) and len(polygons[0]) == 2:
        if verbose == 'debug':
            setting_config("polygons_to_vector: Detected list of (polygon, depth) tuples.") 
        geoms = [g for g, _ in polygons]
        depths = [_d for (_g, _d) in polygons]
    else:
        error("polygons_to_vector expects either a list of geoms or a list of (polygon, depth) tuples.")
        raise ValueError("Invalid polygons input format.")
    
    metadata = {"depth": depths}

    gdf = to_gdf(geoms, metadata, crs)
    
    if affine_val is not None and isinstance(affine_val, Affine):
        gdf = apply_affine_to_gdf(gdf, affine_val)
        info("Applied affine transformation to GeoDataFrame geometries.")
    
    info(f"Converted {len(gdf)} polygons to GeoDataFrame with: CRS {gdf.crs}, shape {gdf.shape}")

    return gdf

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