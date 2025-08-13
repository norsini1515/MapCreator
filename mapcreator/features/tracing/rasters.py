import numpy as np
import rasterio
from rasterio.transform import from_bounds

def _transform_from_extent(width, height, extent):
    return from_bounds(extent["xmin"], extent["ymin"], extent["xmax"], extent["ymax"], width, height)


def init_empty_raster_pair(terrain_path, climate_path, *, width, height, extent, crs):
    transform = _transform_from_extent(width, height, extent)
    profile = dict(driver="GTiff", width=width, height=height, count=1, dtype="uint16", crs=crs, transform=transform, compress="deflate")
    zeros = np.zeros((height, width), dtype="uint16")
    for p in [terrain_path, climate_path]:
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(zeros, 1)