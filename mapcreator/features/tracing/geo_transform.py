"""
Set your continent extent once (e.g., 0..3500 “mi” each way) and everything lands in that world grid. Later you can reproject to a real CRS.
"""

import numpy as np
from affine import Affine

def pixel_affine(width, height, *, xmin=0, ymin=0, xmax=1000, ymax=1000):
    # map pixel (col, row) to world (x, y). Image y grows downward; flip it.
    sx = (xmax - xmin) / width
    sy = (ymax - ymin) / height
    return Affine(sx, 0, xmin, 0, -sy, ymax)  # note -sy, ymax origin at top-left

def apply_affine_to_gdf(gdf, A: "Affine"):
    # apply to all coordinates in-place
    gdf["geometry"] = gdf.geometry.affine_transform((A.a, A.b, A.d, A.e, A.xoff, A.yoff))
    return gdf