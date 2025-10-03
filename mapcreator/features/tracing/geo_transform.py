"""
mapcreator/features/tracing/geo_transform.py

Geo-transform utilities for pixel-to-world coordinate conversion.
Set your continent extent once (e.g., 0..3500 “mi” each way) and everything lands in that world grid. Later you can reproject to a real CRS.
"""
from affine import Affine

def pixel_affine(width, height, *, xmin, ymin, xmax, ymax):
    sx = (xmax - xmin) / width
    sy = (ymax - ymin) / height
    return Affine(sx, 0, xmin, 0, -sy, ymax)

def apply_affine_to_gdf(gdf, A: Affine):
    gdf["geometry"] = gdf.geometry.affine_transform((A.a, A.b, A.d, A.e, A.xoff, A.yoff))
    return gdf
