"""Tracing Pipeline
====================

High–level overview of the image -> vector -> raster flow. This module is the
entry point for turning a hand‑prepared grayscale basemap into geodata layers
and starter rasters.

Processing Steps
----------------
1. Preprocess image (contrast, threshold, optional invert / flood fill) -> 0/1 land mask
2. Detect external contours on land mask -> land polygons (simple shells only)
3. Derive water mask as the inverse of land mask (no need to compute both upstream)
4. Split water mask into:
    - ocean (water connected to image border via flood fill)
    - inland water (remaining water = lakes, seas)
5. Polygonize:
    - ocean: external contours only (no hole handling required)
    - inland: full contour tree -> polygons with holes (islands inside lakes)
6. Merge all classed geometries (land, waterbody, ocean) and dissolve by class
7. Rasterize classes once to produce land_mask (1 = land, 0 = water); derive water_mask by inversion
8. Initialize terrain/climate class rasters as copies of the land mask (paintable later)

Key Simplification
------------------
Land and water masks are mathematical inverses at this stage, so we only
explicitly rasterize land. Water is computed as ``1 - land``. This avoids
drift between two parallel code paths and makes semantics explicit.

Returned Objects
----------------
extract_all() returns: (land_gdf, waterbody_gdf, ocean_gdf, merged_gdf, bin_img)

"""

from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from .img_preprocess import preprocess_image
from .contour_extraction import extract_contour_tree
from .polygonize import land_polygons_from_tree, water_polygons_from_tree
from .gdf_tools import to_gdf, dissolve_class
from .geo_transform import pixel_affine, apply_affine_to_gdf
from .exporters import export_gdf
from .rasters import make_land_water_masks
from .water_classify import split_ocean_vs_inland, inland_mask_to_polygons

from mapcreator.globals.logutil import info, process_step, error
from mapcreator.globals import configs


def _affine(meta):
    return pixel_affine(meta["width"], meta["height"], **meta["extent"])

def land_gdf_from_binary(bin_img: np.ndarray, meta: dict):
    """Build land GeoDataFrame from a binary land mask (1=land, 0=water) using unified contour tree."""
    tree = extract_contour_tree(bin_img > 0)
    polys = land_polygons_from_tree(tree, meta=meta)
    land = to_gdf(polys, {"class": "land"}, crs=meta["crs"]) if polys else gpd.GeoDataFrame(columns=["class", "geometry"], crs=meta["crs"])
    return apply_affine_to_gdf(land, _affine(meta))

def water_gdfs_from_binary(bin_img: np.ndarray, meta: dict):
    """Build ocean + inland water GeoDataFrames from a binary land mask.

    Uses flood-fill to distinguish ocean (connected to image border) from inland water.
    Inland water polygons are built via unified contour tree parity (odd-depth view).
    Ocean polygons remain external shell extraction from flood-filled mask for robustness.
    """
    water_mask = (bin_img == 0).astype("uint8")
    ocean_mask, inland_mask = split_ocean_vs_inland(water_mask)

    # Inland water via unified tree
    inland_tree = extract_contour_tree(inland_mask > 0)
    in_polys = water_polygons_from_tree(inland_tree, meta=meta)

    # Ocean via external shells of ocean_mask (treating all as land-view of ocean_mask)
    ocean_tree = extract_contour_tree(ocean_mask > 0)
    oc_polys = land_polygons_from_tree(ocean_tree, meta=meta)

    ocean = to_gdf(oc_polys, {"class": "ocean"}, crs=meta["crs"]) if oc_polys else gpd.GeoDataFrame(columns=["class", "geometry"], crs=meta["crs"]) 
    waterb = to_gdf(in_polys, {"class": "waterbody"}, crs=meta["crs"]) if in_polys else gpd.GeoDataFrame(columns=["class", "geometry"], crs=meta["crs"]) 

    A = _affine(meta)
    return apply_affine_to_gdf(ocean, A), apply_affine_to_gdf(waterb, A)

def build_merged_base(land_gdf: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame, ocean_gdf: gpd.GeoDataFrame):
    parts = [df for df in [land_gdf, water_gdf, ocean_gdf] if not df.empty]
    if not parts:
        return gpd.GeoDataFrame(columns=["class", "geometry"], crs=land_gdf.crs)
    all_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=land_gdf.crs)
    return dissolve_class(all_gdf, class_col="class")

def extract_all(image: Path, meta: dict):
    """High-level extraction driver.

    Steps:
      1. Preprocess -> binary land mask (1=land)
      2. Land polygons from binary
      3. Ocean + inland water polygons from binary
      4. Merge + dissolve
    Returns (land_gdf, waterbody_gdf, ocean_gdf, merged_gdf, bin_img)
    """
    process_step(f"Extracting features from {image.name}...")
    bin_img = preprocess_image(
        image,
        contrast_factor=meta.get("contrast", 2.0),
        invert=meta.get("invert", False),
        flood_fill=meta.get("flood_fill", False),
    )
    land_gdf = land_gdf_from_binary(bin_img, meta)
    ocean_gdf, waterb_gdf = water_gdfs_from_binary(bin_img, meta)

    info(
        f"Land GDF: {len(land_gdf)} features, Ocean GDF: {len(ocean_gdf)} features, "
        f"Waterbody GDF: {len(waterb_gdf)} features"
    )
    info(
        f"Land GDF CRS: {land_gdf.crs}, Ocean GDF CRS: {ocean_gdf.crs}, "
        f"Waterbody GDF CRS: {waterb_gdf.crs}"
    )
    info(
        f"Land GDF shape {land_gdf.shape=}, Ocean GDF shape {ocean_gdf.shape}, "
        f"Waterbody GDP shape {waterb_gdf.shape}\n"
    )

    process_step("Building merged base...")
    merged_gdf = build_merged_base(land_gdf, waterb_gdf, ocean_gdf)
    return land_gdf, waterb_gdf, ocean_gdf, merged_gdf, bin_img

def write_all(image: Path, out_dir: Path, meta: dict):
    """Run full extraction and write all vector + raster products.

    Returns dict[str, str] of output paths.
    """
    land_gdf, waterb_gdf, ocean_gdf, merged_gdf, _bin = extract_all(image, meta)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Vector exports
    land_path   = export_gdf(land_gdf,   out_dir / configs.LAND_TRACING_EXTRACT_NAME)
    water_path  = export_gdf(waterb_gdf, out_dir / configs.WATER_TRACING_EXTRACT_NAME)
    ocean_path  = export_gdf(ocean_gdf,  out_dir / "ocean.geojson")  # explicit for clarity
    merged_path = export_gdf(merged_gdf, out_dir / configs.MERGED_TRACING_EXTRACT_NAME)

    # Raster exports (land_mask, water_mask, terrain_class, climate_class)
    land_mask_path, water_mask_path, terrain_class_path, climate_class_path = make_land_water_masks(
        merged_gdf, out_dir,
        width=meta["width"], height=meta["height"],
        extent=meta["extent"], crs=meta["crs"]
    )

    return {
        "land": str(land_path),
        "waterbodies": str(water_path),
        "ocean": str(ocean_path),
        "merged": str(merged_path),
        "land_mask": str(land_mask_path),
        "water_mask": str(water_mask_path),
        "terrain": str(terrain_class_path),
        "climate": str(climate_class_path),
    }