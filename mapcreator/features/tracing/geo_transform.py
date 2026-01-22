"""
mapcreator/features/tracing/geo_transform.py

Geo-transform utilities for pixel-to-world coordinate conversion.
Set your continent extent once (e.g., 0..3500 “mi” each way) and everything lands in that world grid. Later you can reproject to a real CRS.
"""
from affine import Affine

from mapcreator.globals.config_models import ExtractConfig

def pixel_affine(width, height, *, xmin, ymin, xmax, ymax):
    sx = (xmax - xmin) / width
    sy = (ymax - ymin) / height
    return Affine(sx, 0, xmin, 0, -sy, ymax)

def apply_affine_to_gdf(gdf, A: Affine):
    gdf["geometry"] = gdf.geometry.affine_transform((A.a, A.b, A.d, A.e, A.xoff, A.yoff))
    return gdf

def affine_from_meta(tracing_cfg:ExtractConfig) -> Affine:
    if tracing_cfg.image_shape is not None:
        w, h = tracing_cfg.image_shape
        return pixel_affine(w, h, 
                            xmin=tracing_cfg.xmin, 
                            xmax=tracing_cfg.xmax, 
                            ymin=tracing_cfg.ymin, 
                            ymax=tracing_cfg.ymax)
    else:
        raise ValueError(f"no 'image_shape'")