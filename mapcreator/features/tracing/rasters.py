"""
mapcreator/features/tracing/rasters.py

Raster creation and initialization from vector data.
This module handles rasterization of land/water masks and initialization
of paintable class rasters (terrain, climate) based on the land mask.

"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from typing import Dict, Tuple
from rasterio.features import rasterize
from pathlib import Path
import yaml
import geopandas as gpd

#package imports
from mapcreator.globals.logutil import info, error, process_step, success, setting_config, warn

def get_default_raster_class_config() -> dict:
    """Return a minimal default raster class configuration.

    The structure mirrors the expected YAML layout used elsewhere:

    - ``classes``: mapping of section name -> {class_label -> class_id}
    - ``colors``:  mapping of section name -> {class_id -> hex_color}

    This default keeps things simple with three high‑level sections
    (``base``, ``terrain``, ``climate``) that all share the same mappings.
    """
    classes_base = {"waterbody": 0, "ocean": 0, "land": 1}
    classes_terrain = classes_base.copy()
    classes_climate = classes_base.copy()
    
    colors_base = {0: "#8BBBEB", 1: "#DDFFBE"}
    colors_terrain = colors_base
    colors_climate = colors_base

    return {
        "classes": {
            "base": classes_base,
            "terrain": classes_terrain,
            "climate": classes_climate,
        },
        "colors": {
            "base": colors_base,
            "terrain": colors_terrain,
            "climate": colors_climate,
        }
    }

#-- RASTERIZATION HELPERS --#
def get_raster_class_config(meta: dict) -> dict:
    """Internal helper to load raster class configuration from YAML path specified in meta."""
    process_step("Loading raster class configurations...")
    cfg_path = meta.get("raster_class_config_path")
    
    if cfg_path is None:
        warn("Raster class config path not specified in meta; using default config.")
        default_cfg_path = Path(__file__).resolve().parents[3] / "config" / "raster_classifications.yml"
        if default_cfg_path.exists():
            cfg_path = str(default_cfg_path.as_posix())
            info(f"Using default raster class config at {cfg_path}")
        else:
            cfg_path = None
            error(f"Default raster class config not found at {default_cfg_path}; proceeding with empty config.")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            class_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        error(f"Failed to load raster classifications YAML at {cfg_path}: {e}")
        class_cfg = {}

    #garantee expected structure
    class_cfg.setdefault("classes", {})
    class_cfg.setdefault("colors", {})
    
    return class_cfg

def _transform_from_extent(width: int, height: int, extent: dict):
    """Build an affine transform from extent bounds and array shape.

    extent: dict with keys xmin, ymin, xmax, ymax (map coordinates)
    """
    return from_bounds(
        extent["xmin"], extent["ymin"],
        extent["xmax"], extent["ymax"],
        width, height,
    )

def _class_id_raster(
    gdf: gpd.GeoDataFrame,
    *,
    width: int,
    height: int,
    extent: dict,
    class_col: str,
    class_to_id: Dict[str, int],
    dtype: str = "uint8",
    verbose: bool = False,
):
    """Rasterize GDF once into integer class ids."""
    if gdf is None or gdf.empty:
        return np.zeros((height, width), dtype=dtype)
    # normalize mapping keys to handle case/spacing
    _map = {str(k).strip().lower(): int(v) for k, v in class_to_id.items()}
    
    if verbose:
        info(f"Rasterizing {len(gdf)} geometries with class mapping: {_map}")

    shapes = (
        (geom, _map.get(str(cls).strip().lower(), 0))
        for geom, cls in zip(gdf.geometry, gdf[class_col])
        if geom is not None and not geom.is_empty
    )
    transform = _transform_from_extent(width, height, extent)

    return rasterize(
        shapes=shapes,
        out_shape=(height, width),
        fill=0,
        transform=transform,
        dtype=dtype,
        all_touched=False,
    )

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b)

def _write_colormapped(path: Path, array: np.ndarray, *, crs, transform, colormap: Dict[int, str] | None, dtype="uint8"):
    profile = {
        "driver": "GTiff",
        "width": array.shape[1],
        "height": array.shape[0],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "photometric": "palette",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        if colormap:
            # keys may come as str from YAML; coerce to int
            cmap = {int(k): _hex_to_rgb(v) for k, v in colormap.items()}
            dst.write_colormap(1, cmap)

 #-- HIGH-LEVEL RASTER CREATION --#

def build_class_raster(
    vector_gdf: gpd.GeoDataFrame,
    out_dir: Path,
    *,
    width: int,
    height: int,
    extent: dict,
    crs,
    class_mapping: Dict[str, int],
    color_mapping: Dict[int, str] | None = None,
    section: str | None = None,
    class_col: str = "class",
    dtype: str = "uint8",
    raster_name: str | None = None,
):
    """Rasterize a single class mapping into one colormapped GeoTIFF.

    Parameters
    ----------
    vector_gdf
        GeoDataFrame containing the geometries and a class column.
    out_dir
        Directory where the output raster will be written.
    width, height
        Raster dimensions in pixels.
    extent
        Dict with keys ``xmin``, ``ymin``, ``xmax``, ``ymax`` describing the
        spatial bounds of the raster in the target CRS.
    crs
        Coordinate reference system for the output raster (anything
        rasterio understands).
    class_mapping
        Mapping from class label (as stored in ``class_col``) to integer id
        to burn into the raster.
    color_mapping
        Optional mapping from integer id to hex color (e.g. ``{"1": "#FF00FF"}``).
        If provided, it is written as a palette colormap on the output.
    section
        Optional human‑readable name for logging and default filename
        construction (for example ``"base"``, ``"terrain"``, ``"climate"``).
    class_col
        Column in ``vector_gdf`` holding class labels to be mapped to
        integer ids.
    dtype
        Numpy dtype for the raster, default ``"uint8"``.
    raster_name
        Optional filename for the output. If omitted, defaults to
        ``f"{section}_class_map.tif"``.

    Returns
    -------
    Path
        Path to the written raster file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = _transform_from_extent(width, height, extent)

    label = section or "<unnamed>"
    setting_config("Using class mapping for section '%s':" % label)
    setting_config(f"  classes: {class_mapping}")
    setting_config(f"  colors: {color_mapping}")

    # Build raster for this section
    arr = _class_id_raster(
        vector_gdf,
        width=width,
        height=height,
        extent=extent,
        class_col=class_col,
        class_to_id=class_mapping,
        dtype=dtype,
    )
    try:
        vals, counts = np.unique(arr, return_counts=True)
        info(f"{label}_class_map values: {dict(zip(vals.tolist(), counts.tolist()))}")
    except Exception:
        pass

    if raster_name is None:
        raster_name = f"{label}_class_map.tif"

    raster_path = out_dir / raster_name

    _write_colormapped(raster_path, arr, crs=crs, transform=transform, colormap=color_mapping, dtype=dtype)

    return raster_path

def make_class_rasters(
    merged_gdf: gpd.GeoDataFrame,
    out_dir: Path,
    *,
    width: int,
    height: int,
    extent: dict,
    crs,
    class_config: Dict | None = None,
    class_col: str = "class",
    dtype: str = "uint8",
):
    """Convenience wrapper to build the standard three class rasters.

    This preserves the previous behavior of returning three rasters
    (world/terrain/climate) while delegating the actual work to
    :func:`build_class_raster`. New code that needs more flexibility can
    call :func:`build_class_raster` directly in a loop with arbitrary
    section names.
    """
    if not class_config:
        class_config = get_default_raster_class_config()

    classes = class_config.get("classes", {})
    colors = class_config.get("colors", {})
    tif_paths = {}
    for class_name, mapping in classes.items():
        if class_name not in colors:
            warn(f"No color mapping found for class '{class_name}'; output will be uncolored.")

        path = build_class_raster(
            merged_gdf,
            out_dir,
            width=width,
            height=height,
            extent=extent,
            crs=crs,
            class_mapping=mapping,
            color_mapping=colors.get(class_name),
            section=class_name,
            class_col=class_col,
            dtype=dtype,
            raster_name=f"{class_name}_class_map.tif",
        )
        info(f"Raster for section '{class_name}' written to: {path}")    
        tif_paths[class_name] = path

    return tif_paths