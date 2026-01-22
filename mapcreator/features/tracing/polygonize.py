"""
mapcreator/features/tracing/polygonize.py

Unified polygonization helpers.

This module now delegates to the unified contour extraction + polygon
assembly logic in ``contour_extraction``. The previous separate functions
``contours_to_polygons`` and ``contours_to_polygons_with_holes`` are retained
as thin wrappers for compatibility but internally reuse shared code.
"""

import geopandas as gpd
from typing import Any, Optional, Tuple, overload, Literal
from collections.abc import Iterator
from affine import Affine
from shapely.ops import unary_union
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.validation import explain_validity

from mapcreator.globals.configs import MIN_POINTS, MIN_AREA
from mapcreator.features.tracing.contour_extraction import ContourTree, _cnt_coords
from mapcreator.globals.logutil import info, process_step, error, warn, setting_config
from mapcreator.features.tracing.geo_transform import apply_affine_to_gdf
from mapcreator.globals.config_models import ExtractConfig

def _extent_polygon(width: int, height: int) -> Polygon:
    # Build extent in PIXEL space (0..width, 0..height). We transform to map coords later.
    process_step("Building extent polygon in pixel space")
    setting_config(f"Extent polygon bounds: (0, 0, {width}, {height})")

    return box(0, 0, width, height)

def _is_shell_of_parity(depth_val: int, build_even: bool) -> bool:
    """True if contour with depth_val should be treated as a shell for this build."""
    return (depth_val % 2 == 0) == build_even

def _collect_holes_for_shell(i: int, tree: ContourTree, *, min_points: int, build_even: bool, verbose: bool|str):
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

def _repair_and_split(poly: Polygon, 
                      *, 
                      min_area: float, 
                      verbose: bool|str = False
                      ) -> Iterator[Polygon]:
    """Fix invalids via buffer(0), split multiparts, yield valid pieces >= min_area."""
    if (not poly.is_valid) or (poly.area < min_area):
        fixed = poly.buffer(0)
        if fixed.is_empty:
            return
        
        if isinstance(fixed, Polygon):
            pieces = [fixed]
        elif isinstance(fixed, MultiPolygon):
            pieces = list(fixed.geoms)
    else:
        pieces = [poly]

    for g in pieces:
        if g.is_valid and g.area >= min_area:
            yield g
        elif verbose in (True, 'debug'):
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
    verbose: bool|str = False,
    include_depth: bool = True,
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
    verbose : bool|str
        Print diagnostics for skipped geometries.

    Returns
    -------
    list
        If include_depth=False (default): list[Polygon]
        If include_depth=True: list[tuple[Polygon, int]] as (geometry, depth) #TODO in the future depth can get wrapped into a ContourMetadata object
        Valid Shapely polygons meet filters. Invalid rings are repaired via buffer(0).
    """

    process_step(f"Assembling {'land' if build_even else 'inland water'} polygons from contours")

    contours = tree.contours
    depth = tree.depth
    children = tree.children

    if verbose in (True, 'debug'):
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
            if include_depth:
                polys.append((g, depth[i]))
            else:
                polys.append(g)

    return polys
        
def build_polygons_from_tree(
    tree: ContourTree,
    tracing_cfg: ExtractConfig,
    *,
    min_area: float = MIN_AREA,
    min_points: int = MIN_POINTS,
    include_depth: bool = True,
) -> tuple[list, list]:
    """Return all polygons (land + inland water), pulling defaults from meta if provided.
       Wrapper around assemble_polygons() to get both even and odd parity contour sets.
    Parameters
    ----------
    tree : ContourTree
        The contour tree.
    tracing_cfg : ExtractConfig
        Extract configuration providing verbosity and optional defaults.
    min_area : float, optional
        Minimum polygon area. If None, pulls from meta or defaults to MIN_AREA.
    min_points : int, optional
        Minimum vertex count in a ring. If None, pulls from meta or defaults to MIN_POINTS.
    include_depth : bool
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

    # Even-depth shells = land polygons
    even_polys = assemble_polygons(
        tree,
        build_even=True,
        min_area=min_area,
        min_points=min_points,
        verbose=tracing_cfg.verbose,
        include_depth=include_depth,
    )

    # Odd-depth shells = inland water polygons
    odd_polys = assemble_polygons(
        tree,
        build_even=False,
        min_area=min_area,
        min_points=min_points,
        verbose=tracing_cfg.verbose,
        include_depth=include_depth,
    )

    return even_polys, odd_polys

@overload
def compute_extent_polygon(
    even_polys: list,
    odd_polys: list,
    tracing_cfg: ExtractConfig,
    *,
    verbose: bool = False,
    include_depth: Literal[True],
) -> Tuple[Polygon | MultiPolygon, int]:
    ...
@overload
def compute_extent_polygon(
    even_polys: list,
    odd_polys: list,
    tracing_cfg: ExtractConfig,
    *,
    verbose: bool = False,
    include_depth: Literal[False] = ...,
) -> Polygon | MultiPolygon:
    ...

def compute_extent_polygon(
    even_polys: list,
    odd_polys: list,
    tracing_cfg: ExtractConfig,
    verbose: bool = False,
    include_depth: bool = True,
) -> Any:
    
    """Compute the extent polygon by subtracting all land and inland water from the image extent box.
    Parameters
    ----------
    even_polys : list[Polygon]
        Land polygons (even-depth shells).
    odd_polys : list[Polygon]
        Inland water polygons (odd-depth shells).
    tracing_cfg : ExtractConfig
        Configuration object containing image_shape and other parameters.
    verbose : bool
        Print diagnostics.
    include_depth : bool
        If True, return a tuple of (geometry, -1) where -1 is the assigned depth for extent/ocean.
    """
    process_step("Computing extent polygon")

    # unwrap (geom, depth) tuples if needed
    if even_polys and isinstance(even_polys[0], tuple):
        even_polys = [p for p, _ in even_polys]
    if odd_polys and isinstance(odd_polys[0], tuple):
        odd_polys = [p for p, _ in odd_polys]

    # Build extent polygon in pixel space (0..width, 0..height)
    if tracing_cfg.image_shape is None:
        raise ValueError("tracing_cfg.image_shape is required to compute extent polygon.")
    width, height = tracing_cfg.image_shape
    extent_poly = _extent_polygon(width, height)

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
    
    if include_depth:
        # Assign depth=-1 for the extent/ocean polygon
        return (extent_poly, -1)

    return extent_poly

def extract_polygons_from_binary(bin_img, 
                                 tracing_cfg: ExtractConfig, 
                                 verbose: bool = False
                                 ) -> tuple[list, list]:
    """Convenience wrapper: from a binary land mask to even/odd polygons.

    Even-depth shells correspond to the foreground class in the mask (1s),
    odd-depth shells to the background holes (0s). This retains the general
    even/odd semantics; mapping to domain labels (land/water) happens upstream.

    Returns (even_polys, odd_polys). If tracing_cfg.compute_extent_polygons is True,
    appends the extent/ocean polygon as (geometry, -1) to the odd list.
    """
    from .contour_extraction import extract_contour_tree

    tree = extract_contour_tree(bin_img > 0, verbose=verbose)
    if verbose:
        info(f"Extracted contours: {len(tree.contours)}")
        if tree.depth:
            info(f"Contour max tree depth: {max(tree.depth)}")

    if not tree.contours:
        return [], []

    # Honor tracing_cfg overrides when provided; otherwise use module defaults
    eff_min_area = tracing_cfg.min_area if tracing_cfg.min_area is not None else MIN_AREA
    eff_min_points = tracing_cfg.min_points if tracing_cfg.min_points is not None else MIN_POINTS

    even_polys, odd_polys = build_polygons_from_tree(
        tree,
        tracing_cfg=tracing_cfg,
        min_area=eff_min_area,
        min_points=eff_min_points,
        include_depth=True,
    )

    if tracing_cfg.compute_extent_polygons:
        extent_poly, extent_depth = compute_extent_polygon(
            even_polys=even_polys,
            odd_polys=odd_polys,
            tracing_cfg=tracing_cfg,
            verbose=verbose,
            include_depth=True,
        )
        if not extent_poly.is_empty:
            odd_polys.append((extent_poly, extent_depth))
            if verbose:
                info(f"Added extent polygon to odd set (depth={extent_depth})")

    return even_polys, odd_polys

def polygons_to_gdf(polygons, crs:str="EPSG:3857", affine_val:Affine|None=None, verbose: bool|str = False) -> gpd.GeoDataFrame:
    """Convert a list of Shapely polygons (or (polygon, depth) tuples) to a GeoDataFrame."""
    from mapcreator.features.tracing.gdf_tools import to_gdf

    # Support both tuples and plain geometries
    if polygons and isinstance(polygons[0], Polygon):
        if verbose == 'debug':
            setting_config("polygons_to_vector: Detected list of Polygon geometries.") 
        geoms = list(polygons)
        depths = [None] * len(geoms)
    elif polygons and isinstance(polygons[0], tuple) and len(polygons[0]) == 2:
        if verbose == 'debug':
            setting_config("polygons_to_vector: Detected list of (polygon, depth) tuples.") 
        geoms = [g for g, _ in polygons]
        depths = [_d for (_g, _d) in polygons]
    else:
        # Handle empty input gracefully
        if not polygons:
            geoms = []
            depths = []
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
    from mapcreator.globals.config_models import ExtractConfig
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

    # Compute extent polygon using image dimensions
    tracing_cfg = ExtractConfig(image_shape=img.shape[::-1], verbose=True, compute_extent_polygons=True)
    extent_poly, extent_depth = compute_extent_polygon(
        even_polys=land_polys,
        odd_polys=water_polys,
        tracing_cfg=tracing_cfg,
        verbose=True,
        include_depth=True,
    )
    if not extent_poly.is_empty:
        print(f"Extent polygon computed with depth={extent_depth} and area={extent_poly.area:.2f}")
    else:
        print("Extent polygon is empty (everything covered by polygons).")
    print('-'*100)