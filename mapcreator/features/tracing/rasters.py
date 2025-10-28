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

#package imports
from mapcreator.globals.logutil import info, error, process_step, success, setting_config

#-- RASTERIZATION HELPERS --#
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
    gdf,
    *,
    width: int,
    height: int,
    extent: dict,
    class_col: str,
    class_to_id: Dict[str, int],
    dtype: str = "uint8",
):
    """Rasterize GDF once into integer class ids."""
    if gdf is None or gdf.empty:
        return np.zeros((height, width), dtype=dtype)
    # normalize mapping keys to handle case/spacing
    _map = {str(k).strip().lower(): int(v) for k, v in class_to_id.items()}
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
def make_class_rasters(
    merged_gdf,
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
    """Create three rasters from merged_gdf using class mappings/colors:
       - world_class_map.tif (base)
       - terrain_class_map.tif (terrain)
       - climate_class_map.tif (climate)

       class_config is expected to have keys 'classes' and 'colors' with nested
       sections 'base', 'terrain', 'climate'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = _transform_from_extent(width, height, extent)

    # Defaults if config not provided
    classes_base = {"waterbody": 0, "ocean": 0, "land": 1}
    classes_terrain = classes_base.copy()
    classes_climate = classes_base.copy()
    colors_base = {0: "#8BBBEB", 1: "#DDFFBE"}
    colors_terrain = colors_base
    colors_climate = colors_base

    if class_config:
        classes = class_config.get("classes", {})
        colors = class_config.get("colors", {})
        classes_base = classes.get("base", classes_base)
        classes_terrain = classes.get("terrain", classes_terrain)
        classes_climate = classes.get("climate", classes_climate)
        colors_base = colors.get("base", colors_base)
        colors_terrain = colors.get("terrain", colors_terrain)
        colors_climate = colors.get("climate", colors_climate)

    setting_config("Using class mappings:")
    setting_config(f"  base: {classes_base}")
    setting_config(f"  terrain: {classes_terrain}")
    setting_config(f"  climate: {classes_climate}")
    setting_config("Using color mappings:")
    setting_config(f"  base: {colors_base}")
    setting_config(f"  terrain: {colors_terrain}")
    setting_config(f"  climate: {colors_climate}")

    # Build rasters
    base_arr = _class_id_raster(
        merged_gdf, width=width, height=height, extent=extent,
        class_col=class_col, class_to_id=classes_base, dtype=dtype
    )
    try:
        vals, counts = np.unique(base_arr, return_counts=True)
        info(f"world_class_map values: {dict(zip(vals.tolist(), counts.tolist()))}")
    except Exception:
        pass
    terrain_arr = _class_id_raster(
        merged_gdf, width=width, height=height, extent=extent,
        class_col=class_col, class_to_id=classes_terrain, dtype=dtype
    )
    try:
        vals, counts = np.unique(terrain_arr, return_counts=True)
        info(f"terrain_class_map values: {dict(zip(vals.tolist(), counts.tolist()))}")
    except Exception:
        pass
    climate_arr = _class_id_raster(
        merged_gdf, width=width, height=height, extent=extent,
        class_col=class_col, class_to_id=classes_climate, dtype=dtype
    )
    try:
        vals, counts = np.unique(climate_arr, return_counts=True)
        info(f"climate_class_map values: {dict(zip(vals.tolist(), counts.tolist()))}")
    except Exception:
        pass

    world_path = out_dir / "world_class_map.tif"
    terrain_path = out_dir / "terrain_class_map.tif"
    climate_path = out_dir / "climate_class_map.tif"

    # Write rasters with colormaps
    _write_colormapped(world_path, base_arr, crs=crs, transform=transform, colormap=colors_base, dtype=dtype)
    _write_colormapped(terrain_path, terrain_arr, crs=crs, transform=transform, colormap=colors_terrain, dtype=dtype)
    _write_colormapped(climate_path, climate_arr, crs=crs, transform=transform, colormap=colors_climate, dtype=dtype)

    return world_path, terrain_path, climate_path