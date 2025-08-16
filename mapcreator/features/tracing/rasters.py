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

# def _transform_from_extent(width, height, extent):
#     xres = (extent["xmax"] - extent["xmin"]) / width
#     yres = (extent["ymax"] - extent["ymin"]) / height
#     return Affine.translation(extent["xmin"], extent["ymax"]) * Affine.scale(xres, -yres)

def _ensure_crs(gdf, crs):
    if gdf.crs is None:
        raise ValueError("merged_gdf.crs is None; set it before rasterizing.")
    if str(gdf.crs) != str(crs):
        # safer to reproject than fail
        return gdf.to_crs(crs)
    return gdf



def make_land_water_masks(merged_gdf, out_dir: Path, *, width, height, extent, crs,
                          dtype="uint8") -> tuple[Path, Path]:
    """
    Writes land_mask.tif (1=land, 0=water) and water_mask.tif (1=water, 0=land).
    Returns (land_mask_path, water_mask_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    process_step(f"Creating land/water masks in {out_dir}...")
    info(f"{width}x{height} pixels, extent: {extent}, crs: {crs}")

    transform = _transform_from_extent(width, height, extent)


    land_shapes  = [(g, 2) for g, c in zip(merged_gdf.geometry, merged_gdf["class"]) if c == "land"]
    water_shapes = [(g, 1) for g, c in zip(merged_gdf.geometry, merged_gdf["class"]) if c in ("waterbody","ocean")]

    land_arr = rasterize(land_shapes,  out_shape=(height, width), transform=transform, fill=1, dtype=dtype, all_touched=False)
    print(f"Land raster shape: {land_arr.shape}, unique values: {np.unique(land_arr)}")
    print(f"Land raster value counts: {np.bincount(land_arr.ravel())}")
    print('land raster made.\n\n')

    water_arr= rasterize(water_shapes, out_shape=(height, width), transform=transform, fill=0, dtype=dtype, all_touched=False)
    print(f"Water raster shape: {water_arr.shape}, unique values: {np.unique(water_arr)}")
    print(f"Water raster value counts: {np.bincount(water_arr.ravel())}")
    print('water raster made.\n\n')

    profile = dict(driver="GTiff", width=width, height=height, count=1,
                   dtype=dtype, crs=crs, transform=transform, compress="deflate",
                   nodata=0)  # explicit: 0 = nodata/background

    land_path  = out_dir / "land_mask.tif"
    water_path = out_dir / "water_mask.tif"
    with rasterio.open(land_path, "w", **profile) as dst: dst.write(land_arr, 1)
    with rasterio.open(water_path, "w", **profile) as dst: dst.write(water_arr, 1)
    
    # class rasters initialized with land_arr (1=land-unclassified, 0=water)
    terrain_class_path = out_dir / "terrain_class_map.tif"
    climate_class_path = out_dir / "climate_class_map.tif"
    with rasterio.open(terrain_class_path, "w", **profile) as dst: dst.write(land_arr, 1)
    with rasterio.open(climate_class_path, "w", **profile) as dst: dst.write(land_arr, 1)


    # optional: write a simple colormap to make it look nice on open
    colormap = {
        1: (65, 105, 225, 255),   # water = royal blue
        2: (200, 200, 200, 255),  # land-unclassified = light gray
    }
    for p in (land_path, water_path, terrain_class_path, climate_class_path):
        try:
            with rasterio.open(p, "r+") as dst: dst.write_colormap(1, colormap)
        except Exception:
            pass  # some stacks don't support colormaps; safe to skip
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