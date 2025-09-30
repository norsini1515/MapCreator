import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.enums import Resampling
from affine import Affine
from pathlib import Path

#package imports
from mapcreator.globals.logutil import info, error, process_step, success


def _transform_from_extent(width, height, extent):
    return from_bounds(extent["xmin"], extent["ymin"],
                       extent["xmax"], extent["ymax"],
                       width, height)

def _ensure_crs(gdf, crs):
    if gdf.crs is None:
        raise ValueError("merged_gdf.crs is None; set it before rasterizing.")
    if str(gdf.crs) != str(crs):
        # safer to reproject than fail
        return gdf.to_crs(crs)
    return gdf


def make_land_water_masks(merged_gdf, out_dir: Path, *, width, height, extent, crs,
                          dtype="uint8"):
    """Create land and water masks plus initial terrain/climate rasters.

    Strategy:
      * Rasterize only land polygons (1=land, 0=water background)
      * Derive water mask as inverse: water = (land == 0) -> 1 else 0
      * Initialize paintable class rasters as copies of land mask

    Returns (land_mask_path, water_mask_path, terrain_class_path, climate_class_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    process_step(f"Creating land/water masks in {out_dir} (single pass)...")
    info(f"{width}x{height} pixels, extent: {extent}, crs: {crs}")

    transform = _transform_from_extent(width, height, extent)

    # Collect land only
    land_shapes = [(g, 1) for g, c in zip(merged_gdf.geometry, merged_gdf["class"]) if c == "land"]

    land_arr = rasterize(
        land_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # 0 = water/background
        dtype=dtype,
        all_touched=False,
    )
    water_arr = (land_arr == 0).astype(dtype)  # inverse

    profile = dict(
        driver="GTiff", width=width, height=height, count=1,
        dtype=dtype, crs=crs, transform=transform, compress="deflate",
        nodata=0
    )

    land_path   = out_dir / "land_mask.tif"
    water_path  = out_dir / "water_mask.tif"
    terrain_class_path = out_dir / "terrain_class_map.tif"
    climate_class_path = out_dir / "climate_class_map.tif"

    with rasterio.open(land_path, "w", **profile) as dst: dst.write(land_arr, 1)
    with rasterio.open(water_path, "w", **profile) as dst: dst.write(water_arr, 1)
    with rasterio.open(terrain_class_path, "w", **profile) as dst: dst.write(land_arr, 1)
    with rasterio.open(climate_class_path, "w", **profile) as dst: dst.write(land_arr, 1)

    # Optional color map
    colormap = {
        0: (65, 105, 225, 255),   # water background (for land mask visualization context)
        1: (200, 200, 200, 255),  # land
    }
    for p in (land_path, terrain_class_path, climate_class_path):
        try:
            with rasterio.open(p, "r+") as dst: dst.write_colormap(1, colormap)
        except Exception:
            pass
    # Provide simple blue/black for water mask
    try:
        with rasterio.open(water_path, "r+") as dst:
            dst.write_colormap(1, {0: (0,0,0,255), 1: (65,105,225,255)})
    except Exception:
        pass

    return land_path, water_path, terrain_class_path, climate_class_path

def init_paintable_class_raster_from_land_mask(land_mask_path, out_path, *, dtype="uint16", nodata=0):
    """
    Create an empty class raster (all NoData) that is *paintable only on land*.
    Copies georeferencing from the land mask and sets the dataset mask accordingly.
    """
    with rasterio.open(land_mask_path) as src:
        mask_valid = src.read(1) > 0
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

    profile = dict(driver="GTiff", width=width, height=height, count=1,
                   dtype=dtype, crs=crs, transform=transform, compress="deflate",
                   nodata=nodata)

    arr = np.full((height, width), nodata, dtype=dtype)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
        # valid pixels = land only
        dst.write_mask(mask_valid.astype("uint8") * 255)
        dst.build_overviews([2,4,8,16], Resampling.nearest)

    return out_path