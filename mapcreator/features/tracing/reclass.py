"""
mapcreator/features/tracing/reclass.py

Utilities to reclassify terrain/climate rasters after initial creation.

Features:
- Burn vector polygons into an existing class raster (terrain/climate/base)
- Reapply colormap from YAML to an existing class raster

Assumptions:
- Class rasters are single-band paletted GTiffs with integer class IDs
- YAML config at config/raster_classifications.yml defines class id and colors per section
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from mapcreator.globals.logutil import info, error, success, process_step
from .rasters import _hex_to_rgb


Section = Union[str, None]


def _load_class_mappings(class_config: Dict, section: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    classes = (class_config or {}).get("classes", {})
    colors = (class_config or {}).get("colors", {})
    class_to_id = classes.get(section, {})
    id_to_color = colors.get(section, {})
    # Normalize id_to_color keys to int (YAML may parse as int already)
    id_to_color = {int(k): str(v) for k, v in id_to_color.items()}
    return class_to_id, id_to_color


def _resolve_class_id(label_or_id: Union[str, int], class_to_id: Dict[str, int]) -> int:
    # If numeric string or int, return as int
    if isinstance(label_or_id, int):
        return int(label_or_id)
    try:
        return int(str(label_or_id))
    except ValueError:
        pass
    # Otherwise interpret as label via mapping (case-insensitive)
    key = str(label_or_id).strip().lower()
    mapping = {str(k).strip().lower(): int(v) for k, v in class_to_id.items()}
    if key not in mapping:
        raise ValueError(f"Unknown class label '{label_or_id}'. Known labels: {list(class_to_id.keys())}")
    return mapping[key]


def apply_palette_from_yaml(raster_path: Path, class_config: Dict, section: str) -> Path:
    """Reapply colormap from YAML to an existing class raster (in place)."""
    raster_path = Path(raster_path)
    _, id_to_color = _load_class_mappings(class_config, section)
    if not id_to_color:
        error(f"No colors configured for section '{section}'. Skipping palette update: {raster_path}")
        return raster_path

    cmap = {int(cid): _hex_to_rgb(hexc) for cid, hexc in id_to_color.items()}
    with rasterio.open(raster_path, "r+") as ds:
        ds.write_colormap(1, cmap)
    success(f"Applied palette for section '{section}' to {raster_path}")
    return raster_path


def burn_polygons_into_class_raster(
    raster_path: Path,
    polygons_path: Path,
    *,
    class_config: Dict,
    section: str,
    label_or_id: Union[str, int],
    output: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Burn polygons into an existing class raster as the given class id/label.

    - raster_path: existing class raster (e.g., terrain_class_map.tif)
    - polygons_path: vector file (GeoJSON/Shapefile) with geometries to assign
    - section: 'terrain' | 'climate' | 'base' (to resolve label->id and palette)
    - label_or_id: class label (string) or ID (int)
    - output: optional output path; if None and overwrite=False, suffix "_edited.tif" is used
    - overwrite: if True, edits are written back to raster_path
    """
    raster_path = Path(raster_path)
    polygons_path = Path(polygons_path)
    class_to_id, id_to_color = _load_class_mappings(class_config, section)
    class_id = _resolve_class_id(label_or_id, class_to_id)

    process_step(f"Burning polygons from {polygons_path.name} into {raster_path.name} as class {class_id} ({label_or_id})")

    # Load raster
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        raster_crs = src.crs

    # Load vectors and align CRS
    gdf = gpd.read_file(polygons_path)
    if gdf.empty:
        raise ValueError(f"No geometries in {polygons_path}")
    if raster_crs is not None and gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    mask = rasterize(
        shapes=shapes,
        out_shape=arr.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    # Assign class
    before_counts = dict(zip(*[x.tolist() for x in np.unique(arr, return_counts=True)]))
    arr[mask] = class_id
    after_counts = dict(zip(*[x.tolist() for x in np.unique(arr, return_counts=True)]))
    info(f"Before counts: {before_counts}")
    info(f"After  counts: {after_counts}")

    # Prepare output path and palette
    if overwrite:
        out_path = raster_path
    else:
        out_path = Path(output) if output else raster_path.with_name(raster_path.stem + "_edited.tif")

    profile.update({"dtype": str(arr.dtype), "count": 1})
    # Maintain paletted GTiff with colormap
    profile["photometric"] = "palette"

    cmap = {int(cid): _hex_to_rgb(hexc) for cid, hexc in id_to_color.items()}
    if class_id not in cmap:
        # Add a default color for the new class if missing (gray)
        cmap[class_id] = (160, 160, 160)
        info(f"Class {class_id} missing in colors for '{section}'. Added default gray; update YAML to customize.")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.write_colormap(1, cmap)

    success(f"Wrote reclassified raster: {out_path}")
    return out_path
