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

#🎯 Step 1: Ray Endpoint Clustering
#🎯 Step 2: Neighborhood Entropy + Mean Distance
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from itertools import product

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from shapely.geometry import Point, LineString, Polygon

from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from mapcreator import directories, summarize_load, world_viewer
from mapcreator.visualization import viewing_util
from mapcreator.map.analysis  import (
    plot_histogram_by_group,
    plot_polar_histogram_by_group,
    plot_scatter_by_group,
    plot_rays
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
    # labels = db.labels_
    
    # end_gdf = gpd.GeoDataFrame({
    #     'geometry': ray_df['end_point'],
    #     'cluster': labels,
    #     'hit_type': ray_df['hit_type'],
    #     'distance': ray_df['distance'],
    #     'angle': ray_df['angle']
    # }, geometry='geometry', crs=None)
    
    return db

def build_feature_matrix(ray_df, use_start=True, use_end=True, use_distance=True, scale=None):
    features = {}
    feature_identifier_str = ""
    if use_end:
        feature_identifier_str += 'endxy'
    
        features["end_x"] = ray_df["end_point"].x
        features["end_y"] = ray_df["end_point"].y
    
    if use_start:
        if(feature_identifier_str != ""):
            feature_identifier_str += "_"
        feature_identifier_str += 'startxy'
        
        features["start_x"] = ray_df["start_point"].x
        features["start_y"] = ray_df["start_point"].y
    
    if use_distance:
        if(feature_identifier_str != ""):
            feature_identifier_str += "_"
        feature_identifier_str += 'dist'
            
        features["distance"] = ray_df["distance"]

    df = pd.DataFrame(features)

    if scale == "minmax":
        scaler = MinMaxScaler()
    elif scale == "standard":
        scaler = StandardScaler()
    else:
        scaler = None

    if scaler:
        if(feature_identifier_str != ""):
            feature_identifier_str += "_"
        feature_identifier_str += scale
        
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df, feature_identifier_str



def summarize_clusters(ray_df: gpd.GeoDataFrame):
    summary = (
        ray_df[ray_df['cluster'] != -1]
        .groupby('cluster')
        .agg(
            point_count=('cluster', 'size'),
            mean_distance=('distance', 'mean'),
            std_distance=('distance', 'std'),
            min_x=('geometry', lambda g: g.bounds.minx.min()),
            max_x=('geometry', lambda g: g.bounds.maxx.max()),
            min_y=('geometry', lambda g: g.bounds.miny.min()),
            max_y=('geometry', lambda g: g.bounds.maxy.max()),
        )
        .reset_index()
    )
    summary['bbox_area'] = (summary['max_x'] - summary['min_x']) * (summary['max_y'] - summary['min_y'])
    summary['density'] = summary['point_count'] / summary['bbox_area']
    return summary.sort_values('point_count', ascending=False)

def plot_clusters(clustered_endpoints, fig=None):
    
    # Ensure cluster is treated as a string (so Plotly uses categorical color)
    clustered_endpoints["cluster_str"] = clustered_endpoints["cluster"].astype(str)
    
    if not fig:
        fig = go.Figure()

    # Loop by cluster to manually assign color and legend entries
    colors = px.colors.qualitative.Dark24
    cluster_ids = clustered_endpoints["cluster_str"].unique()
    color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}
    
    for cid in cluster_ids:
        if cid == -1:
            continue
        sub = clustered_endpoints[clustered_endpoints["cluster_str"] == cid]
        fig.add_trace(go.Scattergl(
            x=sub["x"],
            y=sub["y"],
            mode='markers',
            name=f"Cluster {cid}",
            marker=dict(
                size=6,
                color=color_map[cid],
                opacity=0.7,
                line=dict(width=0)
            ),
            hovertext=[f"Cluster: {row.cluster}, Dist: {row.distance:.1f}, Angle: {row.angle:.2f}" for row in sub.itertuples()],
            hoverinfo="text"
        ))

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title="Ray Endpoint Clusters",
        height=1000,
        width=1000,
        legend_title="Cluster ID",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
        title="Cluster ID",
        x=1.02,  # Shift right of plot
        y=1,
        borderwidth=1,
        bgcolor="white"
        ),
    )

    return fig

def view_cluster_in_world(ocean_gdf, land_gdf, cluster_df):
    cluster_fig = world_viewer.plot_shapes(ocean_gdf)
    cluster_fig = world_viewer.plot_shapes(land_gdf, cluster_fig)
    cluster_fig = plot_clusters(cluster_df, cluster_fig)
    
    return cluster_fig

if __name__ == '__main__':
    input_date = "051025"
    file_path = directories.DATA_DIR / f"ocean_ray_dataset_{input_date}.geojson"
    rays_gdf = load_ray_dataset(file_path).set_crs(None, allow_override=True)
    land_path = directories.SHAPEFILES_DIR / "land_050725.geojson"
    land_gdf = gpd.read_file(land_path)

    ocean_path = directories.SHAPEFILES_DIR / "ocean_050725.geojson"
    ocean_gdf = gpd.read_file(ocean_path)
    #plotting functionality
    if False:
        plot_rays(rays_gdf)

        plot_histogram_by_group(rays_gdf, column='distance', group='hit_type',
                                title="Ray Distance Distribution by Hit Type")
        
        plot_polar_histogram_by_group(rays_gdf, angle_col='angle', group_col='hit_type',
                                    title="Polar Histogram of Ray Angles")

        plot_scatter_by_group(rays_gdf, x='angle', y='distance', group='hit_type',
                            title="Ray Distance vs Angle")
    
    filtered_rays = rays_gdf[rays_gdf['hit_type'] == 'coastline']
    filtered_rays = filtered_rays[filtered_rays['distance'] < filtered_rays['distance'].quantile(.75)]
    
    eps_values = [5, 10, 15, 20]
    min_samples_values = [5, 10, 15]
    use_end_options = [True, False]
    use_start_options = [True, False]
    use_distance_options = [True, False]
    param_grid = [
        combo for combo in product(eps_values, min_samples_values, use_end_options, use_start_options, use_distance_options)
        if any(combo[2:])  # at least one of use_end, use_start, use_distance is True
    ]
    
    rejected_configs = []
    
    for eps, min_samples, use_end, use_start, use_distance in param_grid:
        # Skip degenerate configs
        if not (use_end or use_start or use_distance):
            continue
    
        features, feature_str = build_feature_matrix(
            filtered_rays,
            use_end=use_end,
            use_start=use_start,
            use_distance=use_distance,
            scale=None  # or "minmax"/"standard" if desired later
        )
        
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
        filtered_rays["cluster"] = db.labels_
        filtered_rays["x"] = filtered_rays.end_point.x
        filtered_rays["y"] = filtered_rays.end_point.y
    
        label_count = filtered_rays["cluster"].nunique()
        noise_count = (filtered_rays["cluster"] == -1).sum()
        if noise_count > 650 or label_count < 25:
            print(f"⏭️ Skipping: {feature_str} | eps={eps}, min={min_samples} — Too noisy or too few clusters")
            
            rejected_configs.append({
                "features": feature_str,
                "eps": eps,
                "min_samples": min_samples,
                "clusters": label_count,
                "noise_count": noise_count
            })
            
            continue
        
        print(f"--- {feature_str} | eps={eps}, min={min_samples} ---")
        print(f"Clusters: {label_count - 1}, Noise: {noise_count}")
        
        fig = view_cluster_in_world(ocean_gdf, land_gdf, filtered_rays.copy())
        filename = f"{feature_str}__eps{eps}_min{min_samples}.html"
        viewing_util.save_figure_to_html(fig, directories.DATA_DIR / f"cluster_searchng/{filename}", open_on_export=False)

    
    
    
    
    
    
    
    
    
    
    
    # summary = summarize_clusters(clustered_endpoints)
    # top_clusters = summary.query("point_count > 30 and density > 0.01")  # Tune as needed
    # print(top_clusters[['cluster', 'point_count', 'density']])
    

    # for cid in top_clusters['cluster']:
    #     sub = clustered_endpoints[clustered_endpoints['cluster'] == cid]
    #     fig, ax = plt.subplots()
    #     sub.plot(ax=ax, linewidth=0.4, color='orange')
    #     ax.set_aspect("equal")
    #     ax.invert_yaxis()
    #     plt.title(f"Cluster {cid} Ray Paths")
    #     plt.tight_layout()
        
    #     plt.show()