"""
mapcreator/features/tracing/geo_transform.py

Geo-transform utilities for pixel-to-world coordinate conversion.
Set your continent extent once (e.g., 0..3500 “mi” each way) and everything lands in that world grid. Later you can reproject to a real CRS.
"""
from affine import Affine
import geopandas as gpd

from mapcreator.globals.config_models import ExtractConfig

def pixel_affine(width: int, height: int, *, xmin: float, ymin: float, xmax: float, ymax: float) -> Affine:
    sx = (xmax - xmin) / width
    sy = (ymax - ymin) / height
    return Affine(sx, 0, xmin, 0, -sy, ymax)

def apply_affine_to_gdf(gdf, A: Affine) -> gpd.GeoDataFrame:
    gdf["geometry"] = gdf.geometry.affine_transform((A.a, A.b, A.d, A.e, A.xoff, A.yoff))
    return gdf

def affine_from_meta(tracing_cfg:ExtractConfig) -> Affine:
    if tracing_cfg.image_shape is not None:
        w, h = tracing_cfg.image_shape
        xmin, xmax, ymin, ymax = tracing_cfg.xmin, tracing_cfg.xmax, tracing_cfg.ymin, tracing_cfg.ymax
        
        if xmin is None or xmax is None or ymin is None or ymax is None:
            raise ValueError(
                "missing one of 'xmin', 'xmax', 'ymin', 'ymax':\n"
                f"[{xmin=}, {xmax=}, {ymin=}, {ymax=}]"
            )
        
        return pixel_affine(w, h, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    else:
        raise ValueError(f"no 'image_shape'")