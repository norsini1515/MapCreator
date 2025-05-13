"""
📊 ray_analysis.py

Purpose:
--------
Analyze raycasting results to identify meaningful spatial structures such as:
- Pinch points
- Converging zones
- Ray clusters
- Sealed or semi-enclosed subregions

This module is intended to support map segmentation, interpretation, and feature detection,
without modifying or exporting geometry directly.

Core Functions:
---------------
- analyze_ray_density()
- find_ray_clusters()
- detect_pinches()
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from typing import Optional

from mapcreator import directories, summarize_load

def analyze_ray_density(ray_df: gpd.GeoDataFrame, grid_size: int = 50) -> np.ndarray:
    """
    Breaks the domain into a grid and counts how many rays intersect each cell.

    Returns:
        2D array (heatmap-like) of ray density.
    """
    pass

def detect_pinch_points(ray_df: gpd.GeoDataFrame, threshold: float = 50.0) -> list[Point]:
    """
    Identify pinch points where ray lengths are short or directions converge.

    Returns:
        List of Point locations that may indicate narrow chokepoints.
    """
    pass


def get_rayfile(file_path):
    rays_gdf = gpd.read_file(file_path)

    rays_gdf["start_point"] = rays_gdf.geometry.apply(lambda g: Point(g.coords[0]))
    rays_gdf["end_point"] = rays_gdf.geometry.apply(lambda g: Point(g.coords[-1]))

    summarize_load(rays_gdf)
    print(rays_gdf['hit_type'].value_counts())

    return rays_gdf


if __name__ == '__main__':
    input_date = "051025"
    file_path = directories.DATA_DIR / f"ocean_ray_dataset_{input_date}.geojson"
    rays_gdf = get_rayfile(file_path)