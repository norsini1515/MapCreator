"""
mapcreator/features/tracing/pipeline.py

Tracing Pipeline
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
import sys
from affine import Affine
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from matplotlib import pyplot as plt


from .img_preprocess import process_image
from .polygonize import extract_polygons_from_binary
from .gdf_tools import to_gdf
from .geo_transform import pixel_affine, apply_affine_to_gdf
from .exporters import export_gdf
from .rasters import make_class_rasters
import yaml
from pathlib import Path as _Path
# from .water_classify import split_ocean_vs_inland, inland_mask_to_polygons

from mapcreator.globals.logutil import info, process_step, error
from mapcreator.globals.image_utility import detect_dimensions
from mapcreator import directories as _dirs
from mapcreator.globals import configs


def _affine(meta) -> Affine:
    return pixel_affine(meta["width"], meta["height"], **meta["extent"])

## extract_polygons_from_binary moved to polygonize.py to keep even/odd semantics local to polygonization

def build_merged_base(land_gdf: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    parts = [df for df in [land_gdf, water_gdf] if not df.empty]
    if not parts:
        return gpd.GeoDataFrame(columns=["class", "geometry"], crs=land_gdf.crs)
    all_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=land_gdf.crs)
    return all_gdf #dissolve_class(all_gdf, class_col="class")

def classify_polygons(polygons_with_depths, class_defs: dict, meta: dict) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame for a set of polygons with depth using provided class definitions.

    - polygons_with_depths: expected as [(geom, depth), ...], but will accept [geom, ...] and set depth=None
    - class_defs: base attributes to apply (e.g., {"class": "land"}); any provided 'depth' will be overridden per feature
    - meta: should contain 'crs'
    """
    process_step(f"Classifying {len(polygons_with_depths)} polygons with defs {class_defs}")

    columns = list(class_defs.keys()) + ["geometry"]
    if not polygons_with_depths:
        return gpd.GeoDataFrame(columns=columns, crs=meta.get("crs"))

    # Support both tuples and plain geometries
    if isinstance(polygons_with_depths[0], tuple) and len(polygons_with_depths[0]) == 2:
        geoms = [g for (g, _d) in polygons_with_depths]
        depths = [_d for (_g, _d) in polygons_with_depths]
    else:
        geoms = list(polygons_with_depths)
        depths = [None] * len(geoms)

    # Merge class defs and inject depth list
    class_metadata = dict(class_defs or {})
    class_metadata["depth"] = depths
    if "class" not in class_metadata:
        class_metadata["class"] = "unknown"

    gdf = to_gdf(geoms, class_metadata, crs=meta.get("crs"))
    # Ensure expected column order when possible
    existing = [c for c in columns if c in gdf.columns]
    gdf = gdf[existing + [c for c in gdf.columns if c not in existing]]
    return gdf

def classify_and_transform(polygons_with_depths, class_defs: dict, meta: dict) -> gpd.GeoDataFrame:
    """Classify polygons and apply pixel->map affine in one step."""
    gdf = classify_polygons(polygons_with_depths, class_defs, meta)
    if not gdf.empty:
        gdf = apply_affine_to_gdf(gdf, _affine(meta))
    return gdf

def image_to_binary(image_path: Path, meta: dict, verbose: bool = False) -> np.ndarray:
    """High-level image preprocessing to binary land mask."""
    
    outline_out = _dirs.TEST_DATA_DIR / f"{image_path.stem}_extracted.png"

    land_mask, filled_path = process_image(
        src_path=image_path,
        out_path=outline_out,
        verbose=verbose,
    )
    return land_mask, filled_path

def vectorize_image_to_gdfs(
        image: Path, 
        meta: dict,
        even_defs: dict = configs.LAND_DEFS,
        odd_defs: dict = configs.WATERBODY_DEFS,
        ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """High-level extraction driver.

    Steps:
      1. Preprocess -> binary land mask (1=land)
      2. Land polygons from binary
      3. Ocean + inland water polygons from binary
      4. Merge + dissolve
    Returns (land_gdf, waterbody_gdf, ocean_gdf, merged_gdf, bin_img)
    """
    process_step(f"Extracting even and odd features from {image.name}...")
    
    #indicates if verbose logging is on
    verbose = meta.get("verbose", False)
    if 'width' not in meta or 'height' not in meta:
        w, h = detect_dimensions(image)
        meta["width"] = w
        meta["height"] = h
  

    # Step 1: Process image to centerline outline and filled land mask
    process_step("Step 1: Processing image to outline + filled land mask...")
    land_mask, _filled_path = image_to_binary(image, meta, verbose=verbose)
    bin_img = land_mask
    
    # Step 2: Extract the even and odd contours from the binary image
    process_step("Step 2: Extracting even and odd contours from binary image...")
    even_polys, odd_polys = extract_polygons_from_binary(bin_img=bin_img, meta=meta, verbose=verbose)
    
    if not even_polys or not odd_polys:
        raise ValueError("No even (land) or odd (water) polygons extracted; check input image and preprocessing settings.")
    if verbose:
        info(f"Even (land-view) Polygons: {len(even_polys)}")
        info(f"Odd (water-view) Polygons: {len(odd_polys)}")

    # Step 3: Classify polygons into land / waterbody GDFs and transform
    process_step("Step 3: Building land and waterbody GeoDataFrames...")
    even_gdf = classify_and_transform(even_polys, even_defs, meta)
    odd_gdf = classify_and_transform(odd_polys, odd_defs, meta)
    
    if verbose:
        info(f"\nLand GDF CRS: {even_gdf.crs}, shape {even_gdf.shape}")
        info(f"Water GDF CRS: {odd_gdf.crs}, shape {odd_gdf.shape}")

    process_step("Building merged base...")
    merged_gdf = build_merged_base(even_gdf, odd_gdf)
    info(f"Merged GDF: {len(merged_gdf)} features, CRS: {merged_gdf.crs}, shape {merged_gdf.shape}\n")
    merged_gdf.plot(column="class", legend=True)
    if verbose == 'debug':
        plt.show()
    
    return even_gdf, odd_gdf, merged_gdf

def extract_all(
        image: Path, 
        out_dir: Path, 
        meta: dict, 
        *
        output_vectors: bool = True,
        even_defs: dict = configs.LAND_DEFS,
        odd_defs: dict = configs.WATERBODY_DEFS,
        ) -> dict[str, str]:
    """Run full extraction and write all vector + raster products.

    Returns dict[str, str] of output paths.
    """
    process_step(f"Starting full extraction pipeline for {image.name}...")
    meta['width'], meta['height'] = detect_dimensions(image)

    even_gdf, odd_gdf, merged_gdf = vectorize_image_to_gdfs(image, meta, even_defs=even_defs, odd_defs=odd_defs)
    # sys.exit("Exiting early for debugging (in write_all)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vector exports
    if output_vectors:
        #default to land/waterbody filenames (base case) if not specified in defs
        even_path   = export_gdf(even_gdf,   out_dir / even_defs.get("file_name", configs.LAND_TRACING_EXTRACT_NAME))
        odd_path  = export_gdf(odd_gdf, out_dir / odd_defs.get("file_name", configs.WATER_TRACING_EXTRACT_NAME))
        merged_path = export_gdf(merged_gdf, out_dir / configs.MERGED_TRACING_EXTRACT_NAME)

    # Raster exports: three class rasters using YAML config (base/terrain/climate)
    process_step("Building class rasters...")
    cfg_path = meta.get("raster_class_config_path")
    if cfg_path is None:
        # default: project_root/config/raster_classifications.yml
        cfg_path = str((_Path(__file__).resolve().parents[3] / "config" / "raster_classifications.yml").as_posix())
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            class_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        error(f"Failed to load raster classifications YAML at {cfg_path}: {e}")
        class_cfg = {}

    #!TODO: allow output paths to be customized, terrain and climate are identical, redeundant, specify which raster to output
    #remove output from construction of rasters, return files and data.
    world_class_path, terrain_class_path, climate_class_path = make_class_rasters(
        merged_gdf, out_dir,
        width=meta["width"], height=meta["height"],
        extent=meta["extent"], crs=meta["crs"],
        class_config=class_cfg,
    )
    #all the returns 
    return {
        "even": str(even_path),
        "odd": str(odd_path),
        "merged": str(merged_path),
        "world": str(world_class_path),
        "terrain": str(terrain_class_path),
        "climate": str(climate_class_path),
    }