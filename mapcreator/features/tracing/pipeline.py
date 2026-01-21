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
from .rasters import build_class_raster

from mapcreator.globals.logutil import info, process_step, error, setting_config, success, warn
from mapcreator.globals.image_utility import detect_dimensions
from mapcreator import directories as _dirs
from mapcreator.globals import configs
from mapcreator.globals.config_models import ClassConfig, ExtractConfig, read_config_file

def _validate_output_dir_meta(
        meta: dict, 
        out_dir: Path | str | None,
        test_data_default_subfolder: str,
        ) -> Tuple[Path | None, bool]:
    """Internal helper to validate output directory and determine if outputs should be written.

    Doesn't matter if out_dir is in meta, if not provided as an argument and verbose is not on, no outputs will be written. 
    If verbose is on and out_dir is not provided, defaults to test data directory with a subfolder for the specific step. 

    """
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

# --- Image preprocessing portion of the pipeline --- #
def extract_image(
    image: Path,
    tracing_cfg: ExtractConfig,
    *,
    out_dir: Path = None,
    outline_suffix: str = "_outline",
) -> np.ndarray:
    """Preprocess a hand‑drawn basemap into a binary land mask.

    This is a focused, image‑only front‑end around :func:`process_image` that
    prepares the grayscale input, thresholds it, and returns a 0/1 land mask.

    Depending on ``meta['verbose']`` and ``out_dir``, it can also write
    diagnostic PNGs of the traced outline and filled mask.

    Parameters
    ----------
    image
        Path to the hand‑prepared source basemap image.
    meta
        Metadata dictionary for the pipeline run. May contain ``"verbose"``
        to control logging and output. If ``"image_shape"`` is missing, the
        image dimensions are detected and stored in this dict.
    out_dir
        Optional output directory for diagnostic PNGs. If omitted but
        ``meta['verbose']`` requests output, a test‑data subfolder is chosen
        automatically via ``_validate_output_dir_meta``.
    outline_suffix
        Suffix appended to the image stem for the outline PNG filename. The
        filled version uses the same stem with ``"_filled"`` appended.

    Returns
    -------
    np.ndarray
        2D binary land‑mask array where 1 indicates land and 0 indicates
        water, suitable for downstream polygon extraction.
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
# --- Image preprocessing portion of the pipeline is done --- #


# --- Vector portion of the pipeline --- #
def extract_vectors(
        img: Union[np.ndarray, Path], #either a mask array or a path to an image
        tracing_cfg: ExtractConfig,
        *,
        out_dir: Path = None,
        add_parity: bool = True,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Extract land/water polygons from a binary mask or image.

    This step converts a preprocessed basemap (or its binary land mask) into two
    GeoDataFrames containing the even and odd polygon sets used downstream in
    the tracing pipeline.

    Parameters
    ----------
    img
        Either a 2D NumPy array representing the binary land mask (1 = land,
        0 = water) or a Path to the original source image. If a Path is
        provided, :func:`extract_image` is called internally to build the mask.
    meta
        Metadata dictionary for the pipeline run. Expected to contain entries
        such as ``crs``, ``extent``, ``image_shape``, and optional
        ``verbose`` flags used for logging and transforms.
    out_dir
        Optional directory where intermediate vector outputs will be written.
        When provided (or when ``meta['verbose']`` enables output), two
        GeoJSON files are created: ``even.geojson`` and ``odd.geojson``.
    add_parity
        If True, adds a ``parity`` column to each GeoDataFrame with values
        ``"even"`` and ``"odd"`` respectively.

    Returns
    -------
    Tuple[GeoDataFrame, GeoDataFrame]
        ``(even_gdf, odd_gdf)`` where ``even_gdf`` represents the land-view
        polygons and ``odd_gdf`` represents the water-view polygons in the
        chosen coordinate reference system.

    Raises
    ------
    ValueError
        If no even or odd polygons can be extracted from the binary image.
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
        land_mask = extract_image(image, meta)
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
    """Attach class metadata columns to a vector layer.

    This is a lightweight helper that takes a GeoDataFrame and a dictionary of
    attribute definitions (for example, land/water class tags) and returns a
    copy with those attributes applied as columns.

    Parameters
    ----------
    gdf
        Input GeoDataFrame whose geometries should be labeled.
    class_def
        Mapping of attribute names to values to assign to every row in the
        output GeoDataFrame. Common keys include ``"class"`` and other
        semantic tags used downstream.
    verbose
        If True, emits a warning via the logging utilities when ``"class"`` is
        missing from ``class_def`` and defaults it to ``"unknown"``.

    Returns
    -------
    GeoDataFrame
        A copy of ``gdf`` with one column per key in ``class_def`` and the
        corresponding constant values applied to all features.
    """
    classified_gdf = gdf.copy()
    class_def = dict(class_def) #make a copy to avoid mutating input

    if "class" not in class_def:
            if verbose:
                warn("No class specified in definition; defaulting to 'unknown'.")
            class_def["class"] = "unknown"

    for key, value in class_def.items():
        classified_gdf[key] = value

    return classified_gdf
# --- Vector portion of the pipeline is done ---#


"""Class configuration helpers."""


def load_class_config(meta: ExtractConfig) -> ClassConfig:
    """Load class/label/color configuration using path from ``meta``.

    This now delegates to :func:`read_config_file` so that callers
    receive a typed :class:`ClassConfig` instance instead of a raw
    dictionary.
    """
    process_step("Loading class configuration (labels/classes/colors)...")

    cfg_path = meta.class_config_path
    return read_config_file(cfg_path, kind="class")  # type: ignore[return-value]


def get_even_odd_configs(class_cfg: ClassConfig) -> tuple[dict, dict]:
    """Derive even/odd vector label definitions from config ``labels``.

    Expects ``labels.base.even`` and ``labels.base.odd`` to contain the
    class keys to use for the even/odd polygon sets. If these are missing,
    falls back to :mod:`configs` LAND/WATER defaults.
    """
 
    labels = class_cfg.labels or {}
    base_labels = labels.get("base", {}) if isinstance(labels, dict) else {}

    even_key = base_labels.get("even")
    odd_key = base_labels.get("odd")

    if not even_key or not odd_key:
        warn("labels.base.even/odd not fully specified; falling back to configs.LAND_DEFS/WATER_DEFS.")
        return configs.LAND_DEFS, configs.WATER_DEFS

    even_defs = {"class": even_key}
    odd_defs = {"class": odd_key}

    info(f"Using config-defined vector classes: even -> '{even_key}', odd -> '{odd_key}'.")
    return even_defs, odd_defs

# --- Raster portion of the pipeline --- #
def extract_rasters(
    source: Union[gpd.GeoDataFrame, Path],
    out_dir: Path,
    tracing_cfg,
    class_config: ClassConfig | None = None,
    even_cfg: dict | None = None,
    odd_cfg: dict | None = None,
    *,
    add_parity: bool = True,
) -> dict[str, str]:
    """Run raster extraction for a merged vector layer or directly from an image.

    This helper can be driven either by a precomputed merged GeoDataFrame
    (typically land+water classes) or by an image path, in which case it
    will perform vector extraction and labeling before rasterization.

    It returns paths to the standard world/terrain/climate class rasters.
    For more flexible use (custom sections, filenames, etc.), call
    :func:`build_class_raster` directly in a loop.
    """
    verbose = meta.get("verbose", False)

    # Load class configuration once for this call
    if class_config is None:
        class_config = load_class_config(tracing_cfg)

    if isinstance(source, Path):
        image = source
        process_step(f"Starting raster extraction from image {image.name}...")
        # Vector extraction from image
        even_cfg, odd_cfg = get_even_odd_configs(class_config)
        even_gdf, odd_gdf = extract_vectors(image, meta, out_dir=out_dir, add_parity=add_parity)
        even_gdf = label_vectors(even_gdf, even_cfg, verbose=verbose)
        odd_gdf = label_vectors(odd_gdf, odd_cfg, verbose=verbose)
        merged_gdf = merge_gdfs([even_gdf, odd_gdf], verbose=verbose)
    elif isinstance(source, gpd.GeoDataFrame):
        merged_gdf = source
        process_step("Starting raster extraction from merged GeoDataFrame...")
    else:
        raise ValueError("Source must be either a Path to an image or a merged GeoDataFrame.")

    out_dir.mkdir(parents=True, exist_ok=True)

    process_step("Building class rasters...")
    raster_paths: dict[str, str] = {}
    class_values = class_config.classes or {}
    class_colors = class_config.colors or {}

    for section_name, mapping in class_values.items():
        colors = class_colors.get(section_name, {})
        if not colors:
            warn(f"No color mapping found for section '{section_name}'; defaulting to empty colormap.")

        raster_path = build_class_raster(
            merged_gdf,
            out_dir,
            width=meta["image_shape"][0],
            height=meta["image_shape"][1],
            extent=meta["extent"],
            crs=meta["crs"],
            class_mapping=mapping,
            color_mapping=colors,
            section=section_name,
            class_col="class",
        )
        raster_paths[section_name] = str(raster_path)

    return raster_paths

# --- Raster portion of the pipeline is done --- #

# --- Full pipeline orchestration --- #
def extract_all(
    image: Path,
    tracing_cfg: ExtractConfig,
    *,
    out_dir: Path|None = None,
    add_parity: bool = True,
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

    # Load configuration and derive default even/odd defs if not provided
    class_cfg = load_class_config(tracing_cfg)
    even_cfg, odd_cfg = get_even_odd_configs(class_cfg)

    even_gdf, odd_gdf = extract_vectors(image, meta, out_dir=out_dir, add_parity=add_parity)

    even_gdf = label_vectors(even_gdf, even_cfg, verbose=verbose)
    odd_gdf  = label_vectors(odd_gdf, odd_cfg, verbose=verbose)
    
    merged_gdf = merge_gdfs([even_gdf, odd_gdf], verbose=verbose)

    raster_paths = extract_rasters(merged_gdf, out_dir, meta, class_config=class_cfg, even_cfg=even_cfg, odd_cfg=odd_cfg)

    # For now we only report raster outputs; vector files (even/odd) are
    # written by extract_vectors when outputs are enabled.
    return raster_paths