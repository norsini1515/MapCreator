"""
ray_analysis.py

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
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString, Polygon
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from mapcreator import directories, summarize_load
from mapcreator.map.analysis  import (
    plot_histogram_by_group,
    plot_polar_histogram_by_group,
    plot_scatter_by_group
)

def analyze_ray_density(
    ray_df: gpd.GeoDataFrame,
    grid_size: int = 50,
    bounds: Optional[tuple] = None,
    return_gdf: bool = False
):
    """
    Breaks the domain into a grid and counts how many rays intersect each cell.

    Args:
        ray_df: GeoDataFrame containing ray geometries.
        grid_size: Size of each square cell in map units.
        bounds: Optional bounds override (minx, miny, maxx, maxy).
        return_gdf: If True, returns a GeoDataFrame with cell polygons and counts.

    Returns:
        If return_gdf: GeoDataFrame with 'geometry' and 'count'.
        Else: 2D numpy array (heatmap-like) of ray density.
    """
    from shapely.strtree import STRtree  # faster spatial indexing
    from shapely.geometry.base import BaseGeometry

    if bounds is None:
        bounds = ray_df.total_bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds

    x_steps = np.arange(minx, maxx + grid_size, grid_size)
    y_steps = np.arange(miny, maxy + grid_size, grid_size)

    cell_polys = []
    for x in x_steps[:-1]:
        for y in y_steps[:-1]:
            cell_polys.append(Polygon([
                (x, y),
                (x + grid_size, y),
                (x + grid_size, y + grid_size),
                (x, y + grid_size)
            ]))

    cell_tree = STRtree(cell_polys)
    ray_geoms = list(ray_df.geometry)

    # Count intersections
    cell_counts = {id(cell): 0 for cell in cell_polys}
    for ray in ray_geoms:
        if not isinstance(ray, BaseGeometry):
            continue  # or log warning
        hits = cell_tree.query(ray)
        for cell in hits:
            if isinstance(cell, BaseGeometry) and ray.intersects(cell):
                cell_counts[id(cell)] += 1

    # Convert to grid or GeoDataFrame
    if return_gdf:
        cell_gdf = gpd.GeoDataFrame({
            "geometry": cell_polys,
            "count": [cell_counts[id(cell)] for cell in cell_polys]
        }, geometry="geometry", crs=ray_df.crs)
        return cell_gdf

    else:
        grid = np.zeros((len(y_steps) - 1, len(x_steps) - 1), dtype=int)
        for idx, cell in enumerate(cell_polys):
            xi = int((cell.bounds[0] - minx) // grid_size)
            yi = int((cell.bounds[1] - miny) // grid_size)
            grid[yi, xi] = cell_counts[id(cell)]
        return grid

def load_ray_dataset(ray_file_path):
    print(f'Loading {ray_file_path.stem}. . .')
    rays_gdf = gpd.read_file(ray_file_path)

    rays_gdf["start_point"] = rays_gdf.geometry.apply(lambda g: Point(g.coords[0]))
    rays_gdf["end_point"] = rays_gdf.geometry.apply(lambda g: Point(g.coords[-1]))

    summarize_load(rays_gdf)
    print(rays_gdf['hit_type'].value_counts())

    return rays_gdf

def find_ray_clusters(ray_df: gpd.GeoDataFrame, eps: float = 40, min_samples: int = 10):
    """
    Cluster ray endpoints using DBSCAN to identify dense zones of ray termination.

    Returns:
        GeoDataFrame of end points with a 'cluster' label.
    """
    coords = np.array([[pt.x, pt.y] for pt in ray_df['end_point']])
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    
    end_gdf = gpd.GeoDataFrame({
        'geometry': ray_df['end_point'],
        'cluster': labels,
        'hit_type': ray_df['hit_type'],
        'distance': ray_df['distance'],
        'angle': ray_df['angle']
    }, geometry='geometry', crs=ray_df.crs)
    
    return end_gdf

if __name__ == '__main__':
    input_date = "051025"
    file_path = directories.DATA_DIR / f"ocean_ray_dataset_{input_date}.geojson"
    rays_gdf = load_ray_dataset(file_path).set_crs(None, allow_override=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    rays_gdf.plot(ax=ax, linewidth=0.3, color='orange', alpha=0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Flip it upright for image-style view
    plt.title("Ray Paths")
    plt.tight_layout()
    plt.show()
    #🎯 Step 1: Ray Endpoint Clustering
    #🎯 Step 2: Neighborhood Entropy + Mean Distance
    
    # plot_histogram_by_group(rays_gdf, column='distance', group='hit_type',
    #                         title="Ray Distance Distribution by Hit Type")
    
    # plot_polar_histogram_by_group(rays_gdf, angle_col='angle', group_col='hit_type',
    #                               title="Polar Histogram of Ray Angles")

    # plot_scatter_by_group(rays_gdf, x='angle', y='distance', group='hit_type',
    #                       title="Ray Distance vs Angle")
    
    
    
    if False:
        # Step 1: Focus only on 'coastline' hits (skip boundary rays)
        filtered = rays_gdf[rays_gdf['hit_type'] == 'coastline']
        
        # Step 2: Optional: only short-distance rays (likely inside features)
        filtered = filtered[filtered['distance'] < filtered['distance'].quantile(0.5)]
        
        # Step 3: Then apply clustering on filtered['end_point']
        clustered_endpoints = find_ray_clusters(filtered, eps=15, min_samples=5).set_crs(None, allow_override=True)
        
        print(len(clustered_endpoints['cluster'].unique()))
    
        print(clustered_endpoints['cluster'].value_counts())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        clustered_endpoints.plot(ax=ax, column='cluster', categorical=True, legend=True, markersize=4, cmap='tab20')
        ax.invert_yaxis()
        plt.title("Ray Endpoint Clusters")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    
    filtered_rays_with_clusters = filtered.copy()
    filtered_rays_with_clusters['cluster'] = clustered_endpoints['cluster'].values
    for cid in clustered_endpoints['cluster'].unique():
        if cid == -1:
            continue
        sub = filtered_rays_with_clusters[filtered_rays_with_clusters['cluster'] == cid]
        fig, ax = plt.subplots()
        sub.plot(ax=ax, color='orange', linewidth=0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        plt.title(f"Rays Terminating in Cluster {cid}")
        plt.show()