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
from typing import Tuple, Union
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from matplotlib import pyplot as plt

from .img_preprocess import process_image
from .polygonize import extract_polygons_from_binary, polygons_to_gdf
from .geo_transform import affine_from_meta
from .exporters import export_gdf
from .gdf_tools import merge_gdfs
from .rasters import make_class_rasters
import yaml
from pathlib import Path as _Path
# from .water_classify import split_ocean_vs_inland, inland_mask_to_polygons

from mapcreator.globals.logutil import info, process_step, error, setting_config, success, warn
from mapcreator.globals.image_utility import detect_dimensions
from mapcreator import directories as _dirs
from mapcreator.globals import configs

def _validate_output_dir_meta(
        meta: dict, 
        out_dir: Path | str | None,
        test_data_default_subfolder: str,
        ) -> Tuple[Path | None, bool]:
    """Internal helper to validate output directory and determine if outputs should be written."""
    verbose = meta.get("verbose", False)
    write_outputs = (out_dir is not None) or verbose in (True, "info", "debug")

    if out_dir is not None:
        if isinstance(out_dir, str):
            out_dir = Path(out_dir).resolve() #

    if verbose in (True, "info", "debug"):
        if out_dir is None:
            setting_config("Verbose Mode is On; Output directory not specified, defaulting to test data directory.")
            out_dir = _dirs.TEST_DATA_DIR / test_data_default_subfolder
            out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir, write_outputs

def extract_image(
    image: Path,
    meta: dict,
    *,
    out_dir: Path = None,
    outline_suffix: str = "_outline",
) -> np.ndarray:
    """
    Preprocess a hand-drawn basemap to a binary land mask image.
    doesnt' output by default unless verbose is on or out_dir specified

    This is a focused, image-only front-end around ``process_image``.

    It writes two PNGs into ``out_dir``:
      - ``<stem><outline_suffix>.png``: centerline outline
      - ``<stem><outline_suffix>_filled.png``: filled land mask (white land, black water)

    Returns filled land mask array.
    """
    process_step(f"Extracting land mask image from {image.name}...")
    
    verbose = meta.get("verbose", False)
    out_dir, write_outputs = _validate_output_dir_meta(meta, out_dir, 'extract_images')

    if write_outputs:
        info(f"Output files will be written to {out_dir}")
        outline_path = out_dir / f"{image.stem}{outline_suffix}.png"
    else:
        outline_path = None

    #Step 0: Ensure we have image dimensions in metadata
    if 'image_shape' not in meta:
        info(f"Dimensions not found in metadata; detecting from image...")
        meta["image_shape"] = detect_dimensions(image)

    land_mask, _ = process_image(
        src_path=image,
        out_path=outline_path,
        verbose=verbose,
        output_file=write_outputs,
    )

    return land_mask

def extract_vectors(
        img: Union[np.ndarray, Path], #either a mask array or a path to an image
        meta: dict,
        *,
        out_dir: Path = None,
        add_parity: bool = True,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Run vector extraction only and write GeoDataFrames to disk.

    """
    verbose = meta.get("verbose", False)
    out_dir, write_outputs = _validate_output_dir_meta(meta, out_dir, 'extract_vectors')

    if isinstance(img, Path):
        image = img
        source=image.name
    else:
        bin_img = img
        image = None
        source='array'
    process_step(f"Extracting vectors from {source}...")
    
    #-----------    
    if image is not None:
        # Step 1: Process image to centerline outline and filled land mask
        process_step("Processing image to outline + filled land mask...")
        land_mask = extract_image(image, meta=meta)
        bin_img = land_mask
    
    # Step 2: Extract the even and odd contours from the binary image
    process_step("Step 2: Extracting even and odd contours from binary image...")
    even_polys, odd_polys = extract_polygons_from_binary(bin_img=bin_img, meta=meta, verbose=verbose)
    
    if not even_polys or not odd_polys:
        raise ValueError("No even (land) or odd (water) polygons extracted; check input image and preprocessing settings.")
    if verbose:
        info(f"Even (land-view) Polygons: {len(even_polys)}")
        info(f"Odd (water-view) Polygons: {len(odd_polys)}")
    
    # Step 3: Classify and transform polygons to GeoDataFrames
    process_step("Step 3: Defining even and odd polygon sets...")
    affine_val = affine_from_meta(meta)
    even_gdf = polygons_to_gdf(even_polys, crs=meta.get("crs"), affine_val=affine_val)
    odd_gdf = polygons_to_gdf(odd_polys, crs=meta.get("crs"), affine_val=affine_val)
    
    if add_parity:
        even_gdf["parity"] = "even"
        odd_gdf["parity"] = "odd"
    
    success("Constructed GeoDataFrames from polygons.")

    # Step 4: Export GeoDataFrames if requested
    if write_outputs:
        process_step("Step 4: Exporting GeoDataFrames to files...")
        even_path = out_dir / f"even.geojson"
        odd_path = out_dir / f"odd.geojson"

        export_gdf(even_gdf, even_path, verbose=verbose)
        export_gdf(odd_gdf, odd_path, verbose=verbose)

    return even_gdf, odd_gdf

def label_vectors(
        gdf: gpd.GeoDataFrame,
        class_def: dict,
        verbose: bool = False,
    ) -> gpd.GeoDataFrame:
    """Assign class labels and metadata to a GeoDataFrame based on provided definitions."""
    classified_gdf = gdf.copy()
    class_def = dict(class_def) #make a copy to avoid mutating input

    if "class" not in class_def:
            if verbose:
                warn("No class specified in definition; defaulting to 'unknown'.")
            class_def["class"] = "unknown"

    for key, value in class_def.items():
        classified_gdf[key] = value

    return classified_gdf

def _build_class_rasters(
    merged_gdf: gpd.GeoDataFrame,
    out_dir: Path,
    meta: dict,
) -> dict[str, str]:
    """Internal helper to build world/terrain/climate class rasters.

    Uses the same YAML-driven configuration as the original ``extract_all``.
    """
    process_step("Building class rasters...")
    cfg_path = meta.get("raster_class_config_path")
    if cfg_path is None:
        # default: project_root/config/raster_classifications.yml
        cfg_path = str(
            (_Path(__file__).resolve().parents[3] / "config" / "raster_classifications.yml").as_posix()
        )
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            class_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        error(f"Failed to load raster classifications YAML at {cfg_path}: {e}")
        class_cfg = {}

    #!TODO: allow output paths to be customized, terrain and climate are identical, redeundant, specify which raster to output
    #remove output from construction of rasters, return files and data.
    even_class_path, odd_class_path, merged_class_path = make_class_rasters(
        merged_gdf,
        out_dir,
        width=meta["image_shape"][0],
        height=meta["image_shape"][1],
        extent=meta["extent"],
        crs=meta["crs"],
        class_config=class_cfg,
    )

    return {
        "world": str(even_class_path),
        "terrain": str(odd_class_path),
        "climate": str(merged_class_path),
    }

def extract_rasters(
    image: Path,
    out_dir: Path,
    meta: dict,
    *,
    even_defs: dict = configs.LAND_DEFS,
    odd_defs: dict = configs.WATERBODY_DEFS,
) -> dict[str, str]:
    """Run raster extraction only (no vector file writing).

    This recomputes polygons from the image and builds world/terrain/climate
    class rasters, returning a dict of raster paths.
    """
    process_step(f"Starting raster extraction for {image.name}...")
    meta["image_shape"] = detect_dimensions(image)

    # We only need the merged GeoDataFrame; vectors are not written here.
    _even_gdf, _odd_gdf, merged_gdf = vectorize_image_to_gdfs(
        image,
        meta,
        even_defs=even_defs,
        odd_defs=odd_defs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    return _build_class_rasters(merged_gdf, out_dir, meta)

def extract_all(
    image: Path,
    meta: dict,
    *,
    out_dir: Path|None = None,
    add_parity: bool = True,
    even_defs: dict = configs.LAND_DEFS,
    odd_defs: dict = configs.WATER_DEFS,
    even_export_defs: dict = configs.LAND_EXPORT_DEFS,
    odd_export_defs: dict = configs.WATER_EXPORT_DEFS,
) -> dict[str, str]:
    """
    Run full extraction and write all vector + raster products.
    The full scope of the image to data pipeline.
    Only writes if out_dir is specified or verbose mode is on.
    """
    process_step(f"Starting full extraction pipeline for {image.name}...")

    verbose = meta.get("verbose", False)
    if verbose:
        info(f"Verbose mode is on.")
        
    out_dir, write_outputs = _validate_output_dir_meta(meta, out_dir, 'full_extraction')
    
    if write_outputs:
        info(f"Output files will be written to {out_dir}")

    even_gdf, odd_gdf = extract_vectors(image, meta, out_dir=out_dir, add_parity=add_parity)
    
    even_gdf = label_vectors(even_gdf, even_defs, verbose=verbose)
    odd_gdf = label_vectors(odd_gdf, odd_defs, verbose=verbose)
    
    merged_gdf = merge_gdfs([even_gdf, odd_gdf], verbose=verbose)
    



    raster_paths = _build_class_rasters(merged_gdf, out_dir, meta)

    # Merge vector + raster paths into a single mapping.
    return {**vec_paths, **raster_paths}