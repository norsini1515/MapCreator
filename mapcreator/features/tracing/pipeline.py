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
from attr import validate
from cv2 import merge
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from matplotlib import pyplot as plt

from .img_preprocess import process_image
from .polygonize import extract_polygons_from_binary, polygons_to_gdf
from .geo_transform import affine_from_meta
from .rasters import build_class_raster

from mapcreator import directories as _dirs
from mapcreator.globals.logutil import info, process_step, error, setting_config, success, warn
from mapcreator.globals.image_utility import detect_dimensions
from mapcreator.globals.gdf_tools import merge_gdfs
from mapcreator.globals import configs, export_gdf, export_gdfs
from mapcreator.globals.config_models import (
    ExtractConfig, ClassConfig, ClassDef,
    read_class_resolver_from_extract
)

def _validate_output_dir_meta(
        tracing_cfg: ExtractConfig, 
        out_dir: Path | str | None,
        test_data_default_subfolder: str,
        ) -> Tuple[Path | None, bool]:
    """Validate output directory and determine whether to write outputs.

    If `out_dir` is missing and verbosity is enabled, default to the test data
    directory plus `test_data_default_subfolder`.
    """
    verbose = tracing_cfg.verbose
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
    out_dir: Path|None = None,
    outline_suffix: str = "_outline",
) -> np.ndarray:
    """Preprocess a hand‑drawn basemap into a binary land mask.

    This is a focused, image‑only front‑end around :func:`process_image` that
    prepares the grayscale input, thresholds it, and returns a 0/1 land mask.

    Depending on tracing configuration verbosity and ``out_dir``, it can also write
    diagnostic PNGs of the traced outline and filled mask.

    Parameters
    ----------
    image
        Path to the hand‑prepared source basemap image.
    tracing_cfg
        Extract configuration for the pipeline run. If ``image_shape`` is
        missing, the image dimensions are detected and stored on this object.
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
    
    verbose = bool(tracing_cfg.verbose)
    out_dir, write_outputs = _validate_output_dir_meta(tracing_cfg, out_dir, 'extract_images')

    if write_outputs:
        info(f"Output files will be written to {out_dir}")
        outline_path = out_dir / f"{image.stem}{outline_suffix}.png"
    else:
        outline_path = None

    #Step 0: Ensure we have image dimensions in metadata
    if tracing_cfg.image_shape is None:
        info("Dimensions not found in config; detecting from image...")
        tracing_cfg.image_shape = detect_dimensions(image)

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
        out_dir: Path|None = None,
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
    tracing_cfg
        Extract configuration for the pipeline run. Expected to expose entries
        such as ``crs``, extent bounds (``xmin``, ``ymin``, ``xmax``, ``ymax``),
        and ``image_shape`` for transform and rasterization.
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
    verbose = tracing_cfg.verbose
    out_dir, write_outputs = _validate_output_dir_meta(tracing_cfg, out_dir, 'extract_vectors')

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
        land_mask = extract_image(image, tracing_cfg)
        bin_img = land_mask
    
    # Step 2: Extract the even and odd contours from the binary image
    process_step("Step 2: Extracting even and odd contours from binary image...")
    even_polys, odd_polys = extract_polygons_from_binary(bin_img=bin_img, tracing_cfg=tracing_cfg)
    
    if not even_polys or not odd_polys:
        raise ValueError("No even (land) or odd (water) polygons extracted; check input image and preprocessing settings.")
    if verbose:
        info(f"Even (land-view) Polygons: {len(even_polys)}")
        info(f"Odd (water-view) Polygons: {len(odd_polys)}")
    
    # Step 3: Classify and transform polygons to GeoDataFrames
    process_step("Step 3: Defining even and odd polygon sets...")
    affine_val = affine_from_meta(tracing_cfg)
    crs_val = tracing_cfg.crs
    even_gdf = polygons_to_gdf(even_polys, crs=crs_val, affine_val=affine_val)
    odd_gdf = polygons_to_gdf(odd_polys, crs=crs_val, affine_val=affine_val)
    
    even_gdf["parity"] = "even" if tracing_cfg.add_parity else "none"
    odd_gdf["parity"] = "odd" if tracing_cfg.add_parity else "none"
    
    success("Constructed GeoDataFrames from polygons.")

    # Step 4: Export GeoDataFrames if requested
    if write_outputs:
        process_step("Step 4: Exporting GeoDataFrames to files...")
        even_path = out_dir / f"vectors/even.geojson"
        odd_path = out_dir / f"vectors/odd.geojson"

        export_gdf(even_gdf, even_path, verbose=tracing_cfg.verbose, add_time_stamp=True)
        export_gdf(odd_gdf, odd_path, verbose=tracing_cfg.verbose, add_time_stamp=True)

    return even_gdf, odd_gdf

def label_vectors(
        gdf: gpd.GeoDataFrame,
        class_def: ClassDef,
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
        Definition of attributes to assign to every row in the
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

    for key, value in class_def.__dict__.items():
        if verbose:
            info(f"Labeling geometries with '{key}': '{value}'")

        classified_gdf[key] = value

    return classified_gdf

def label_and_merge_by_section(
        tracing_cfg: ExtractConfig,
        class_config: ClassConfig|None = None,
        *,
        even_base_gdf: gpd.GeoDataFrame,
        odd_base_gdf: gpd.GeoDataFrame,
    ) -> dict[str, gpd.GeoDataFrame]:

    merged_vector_gdfs: dict[str, gpd.GeoDataFrame] = {}
    verbose = bool(tracing_cfg.verbose)

    if class_config is None:
        class_config = read_class_resolver_from_extract(tracing_cfg) 
    
    for section_classification in class_config.get_run_scheme_sections():
        even_parity = "even" if tracing_cfg.add_parity else "none"
        odd_parity = "odd" if tracing_cfg.add_parity else "none"
        
        even_id, even_def = class_config.resolve(section=section_classification, parity=even_parity)
        odd_id, odd_def = class_config.resolve(section=section_classification, parity=odd_parity)
        
        if verbose:
            info(f"[{section_classification}] even -> {even_id} ({even_def.name})")
            info(f"[{section_classification}]  odd -> {odd_id} ({odd_def.name})")

        even_gdf = label_vectors(even_base_gdf.copy(), even_def, verbose=verbose)
        odd_gdf = label_vectors(odd_base_gdf.copy(), odd_def, verbose=verbose)
        
        merged_vector_gdfs[section_classification] = merge_gdfs([even_gdf, odd_gdf], verbose=verbose)
    
    return merged_vector_gdfs

# --- Vector portion of the pipeline is done ---#

# --- Raster portion of the pipeline --- #
def extract_rasters(
    *,
    source: Union[gpd.GeoDataFrame, Path, dict[str, gpd.GeoDataFrame]],
    out_dir: Path | None,
    tracing_cfg: ExtractConfig,
    class_config: ClassConfig | None = None,
) -> dict[str, str]:
    """Build one class raster per section (base/terrain/climate/...) using ClassResolver.

    Inputs:
      - Path: image path; will extract vectors, label per section, merge, rasterize
      - GeoDataFrame: already-merged vectors; will rasterize once per section? (see below)
      - dict[str, GeoDataFrame]: pre-merged per-section vectors; rasterize each
    """
    verbose = bool(tracing_cfg.verbose)

    out_dir, write_outputs = _validate_output_dir_meta(tracing_cfg, out_dir, 'extract_rasters')
    # Load class configuration once for this call
    if class_config is None:
        class_config = read_class_resolver_from_extract(tracing_cfg) 

    # elif not class_config.validate():
    #     error("Provided class configuration is invalid; aborting raster extraction.")
    #     raise ValueError("Invalid class configuration provided.")
    
    # ------------------------------------------------------------------
    # 1) Build merged_vector_gdfs: dict[section -> merged gdf]
    # ------------------------------------------------------------------
    merged_vector_gdfs: dict[str, gpd.GeoDataFrame] = {}

    if isinstance(source, Path):
        image = source
        process_step(f"Starting raster extraction from image {image.name}...")
        
        # Extract base vectors once
        even_base_gdf, odd_base_gdf = extract_vectors(img=image, tracing_cfg=tracing_cfg, out_dir=out_dir)
        
        #Label per section and merge
        process_step("Labeling and merging vectors per section...")
        merged_vector_gdfs = label_and_merge_by_section(
                                tracing_cfg=tracing_cfg, 
                                class_config=class_config, 
                                even_base_gdf=even_base_gdf, 
                                odd_base_gdf=odd_base_gdf
                            )

    elif isinstance(source, dict):
        # Already have per-section merged GeoDataFrames
        if verbose:
            info(f"Using provided dict of {len(source)} merged GeoDataFrames for rasterization.")
        merged_vector_gdfs = {k: v.copy() for k, v in source.items()}

    elif isinstance(source, gpd.GeoDataFrame):
        # Single merged gdf. Two reasonable behaviors:
        #   A) Rasterize once as "base" only
        #   B) Rasterize same geometry for all sections (not usually meaningful unless class ids match)
        merged_vector_gdfs["base"] = source.copy()

    else:
        raise ValueError("source must be Path, GeoDataFrame, or dict[str, GeoDataFrame].")
    

    # ------------------------------------------------------------------
    # 2) Rasterize per section using registry-derived mappings
    # ------------------------------------------------------------------
    if tracing_cfg.image_shape is None:
        raise ValueError("tracing_cfg.image_shape is required to build rasters")
    
    width, height = tracing_cfg.image_shape
    
    raster_paths: dict[str, str] = {}
    process_step("Building class rasters...")

    for section_name, merged_gdf  in merged_vector_gdfs.items():
        # Build mappings from registry for this section using ClassConfig.
        # class_mapping: class label ("name" column) -> integer id
        # color_mapping: id -> hex color
        try:
            id_to_def = class_config.registry.defines[section_name]
        except KeyError:
            # If the caller passed a dict with sections not in registry, fail loudly
            raise ValueError(f"Section '{section_name}' not found in class registry. "
                             f"Available: {class_config.get_registry_sections()}")

        # Map from human-readable class name to numeric id for rasterization,
        # and from id to color for the output colormap.
        class_mapping = {ddef.name: cid for cid, ddef in id_to_def.items()}
        color_mapping = {cid: ddef.color for cid, ddef in id_to_def.items()}

        if verbose:
            info(f"[{section_name}] class_mapping={class_mapping}")
            info(f"[{section_name}] color_mapping={color_mapping}")

        raster_path = build_class_raster(
            merged_gdf,
            out_dir,
            width=width,
            height=height,
            extent={
                "xmin": tracing_cfg.xmin,
                "ymin": tracing_cfg.ymin,
                "xmax": tracing_cfg.xmax,
                "ymax": tracing_cfg.ymax,
            },
            crs=tracing_cfg.crs,
            class_mapping=class_mapping,
            color_mapping=color_mapping,
            section=section_name,
            # label_vectors stores the resolved class name under the "name"
            # column for each geometry; this is what we map back to ids.
            class_col="name",
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
    # print(tracing_cfg.verbose)
    # print(bool(tracing_cfg.verbose))
    verbose = bool(tracing_cfg.verbose)
    if verbose:
        setting_config(f"Verbose mode is on.")
        
    out_dir, write_outputs = _validate_output_dir_meta(tracing_cfg, out_dir, 'full_extraction')
    if write_outputs:
        info(f"Output files will be written to {out_dir}")
    # ------------------------------------------------------------------
    # Extract vectors
    process_step(f"Extracting base vectors from image {image.name}...")
    # Extract base vectors once
    even_base_gdf, odd_base_gdf = extract_vectors(img=image, tracing_cfg=tracing_cfg, out_dir=out_dir)
    # Load configuration and derive default even/odd defs if not provided
    class_config = read_class_resolver_from_extract(tracing_cfg)
    # Label per section and merge
    merged_vector_gdfs: dict[str, gpd.GeoDataFrame] = {}
    if verbose:
        info(f"class_config.get_run_scheme_sections(): {class_config.get_run_scheme_sections()}")

    process_step("Labeling and merging vectors per section...")
    merged_vector_gdfs = label_and_merge_by_section(
                                tracing_cfg=tracing_cfg, 
                                class_config=class_config, 
                                even_base_gdf=even_base_gdf, 
                                odd_base_gdf=odd_base_gdf
                            )
    
    if write_outputs and out_dir is not None:
        export_gdfs(merged_vector_gdfs, out_dir / "vectors", verbose=tracing_cfg.verbose, add_time_stamp=True)
    else:
        warn("Outputs disabled (no out_dir and verbosity off); skipping rasterization.")
        raster_paths = {}

    # ------------------------------------------------------------------
    # Rasterize per section
    process_step("Rasterizing per section...")
    raster_paths = extract_rasters(source=merged_vector_gdfs, 
                                   out_dir=out_dir/'rasters', 
                                   tracing_cfg=tracing_cfg, 
                                   class_config=class_config)
    
    # For now we only report raster outputs; vector files (even/odd) are
    # written by extract_vectors when outputs are enabled.
    return raster_paths

if __name__ == "__main__":
    print("This module is not intended to be run directly; please use it as part of the MapCreator package.")