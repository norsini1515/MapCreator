"""
🧭 image_map_isolate_ocean_feature_classifier.py

Module Purpose:
---------------
This module is responsible for isolating and classifying water features that are 
connected to the ocean — such as fjords, ravines, and coastal seas — from the main 
ocean polygon. These features are identified from the ocean mask layer generated 
by subtracting landmass from the full map extent.

Key Goals:
----------
1. Generate an ocean mask from the landmass layer.
2. Analyze the ocean polygon to identify candidate regions of interest.
3. Detect "pinched" or enclosed water regions that are likely non-oceanic features.
4. Programmatically place "seals" (dividers) between the ocean and candidate features.
5. Split the ocean polygon using these seals and classify segments into:
   - 'ocean' (still part of the main ocean polygon)
   - 'connected_feature' (coastal sea, ravine, fjord, etc.)
6. Output: A clean, labeled shapefile of ocean-connected features.

Core Steps:
-----------
1. **Ocean Mask Generation**: 
   Use the bounding box minus the landmass to create the "true ocean" geometry.

2. **Boundary Sampling**:
   Sample points along the perimeter of the ocean polygon for analysis.

3. **Neighborhood Analysis**:
   For each sampled point, evaluate its local context using:
   - Pinch Score: Based on spatial proximity and curvature.
   - Ray-Casting: Probe into the water to measure enclosure or openness.

4. **Pinch Detection**:
   Identify local maxima in pinch or constriction scores — these are potential feature dividers.

5. **Seal Placement**:
   Programmatically connect narrow gaps using synthetic "seal" lines that divide features.

6. **Polygon Segmentation**:
   Cut the ocean polygon using seals and relabel each resulting sub-polygon.

7. **Output Classification**:
   Return a GeoDataFrame or shapefile with each region labeled either as 'ocean' or 'connected_feature'.

Future Extensions:
------------------
- Interactive labeling/validation in Dash.
- Semi-supervised ML to refine classification heuristics.
- User-placed seals with automatic refinement.
- Spatial feature embedding for contextual classification.

"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Union
from shapely import geometry
from shapely.geometry import Polygon, MultiPolygon, Point, box, LineString

from mapcreator import world_viewer
from mapcreator import directories, configs
import mapcreator.map.shapefile as shapefiles
from mapcreator.visualization import viewing_util
from mapcreator.scripts.extract_images import get_image_dimensions

def union_shapefile(land_shapefile_paths):
    """
    Reads and unions all geometries from one or more shapefiles into a single-row GeoDataFrame.
    Returns a GeoDataFrame for consistency across the module.
    """
    all_geoms = []

    for path in land_shapefile_paths:
        gdf = gpd.read_file(path)
        all_geoms.append(gdf.geometry)

    combined_series = gpd.GeoSeries(pd.concat(all_geoms, ignore_index=True))
    unioned = combined_series.union_all()

    # Wrap result in a GeoDataFrame for consistency
    return gpd.GeoDataFrame({'geometry': [unioned]}, crs=gdf.crs)

def generate_voided_extent(
    land_shapefile_paths: Union[Path, list[Path]],
    dimensions: tuple = (1200, 1200)
) -> gpd.GeoDataFrame:
    """
    Generate a polygon layer representing the true ocean by subtracting land from the full image extent.

    Args:
        land_shapefile_paths (Path or list[Path]): One or more shapefile paths representing landmasses.
        default_dim (tuple): Default size (width, height) if image not provided.

    Returns:
        GeoDataFrame: GeoDataFrame representing the ocean mask polygon(s).
    """
    if isinstance(land_shapefile_paths, Path):
        land_shapefile_paths = [land_shapefile_paths]
    elif not isinstance(land_shapefile_paths, list):
        raise TypeError("land_shapefile_paths must be a Path or a list of Paths.")
    

    # Get image size or use default
    width, height = dimensions

    # Create outer bounding box representing the full image extent
    world_bbox = box(0, 0, width, height)

    # Subtract land polygons from the bounding box to isolate the ocean
    land_geom  = union_shapefile(land_shapefile_paths).geometry.iloc[0]
    ocean_geom = world_bbox.difference(land_geom)
    
    #------
    print("Ocean Geometry Type:", type(ocean_geom))
    if isinstance(ocean_geom, Polygon):
        print(" - Exterior length:", len(ocean_geom.exterior.coords))
        print(" - Number of holes:", len(ocean_geom.interiors))
    elif isinstance(ocean_geom, MultiPolygon):
        for i, poly in enumerate(ocean_geom.geoms):
            print(f" - Polygon {i}: {len(poly.exterior.coords)} exterior coords, {len(poly.interiors)} holes")
    #------
    ocean_gdf = gpd.GeoDataFrame({'geometry': [ocean_geom]}, crs="EPSG:4326")
    ocean_gdf["type"] = "ocean"
    ocean_gdf["source"] = ";".join(p.name for p in land_shapefile_paths)

    return ocean_gdf

def sample_perimeter_points(ocean_df: gpd.GeoDataFrame, spacing: int = 15, sampling_method: int = 1) -> list[Point]:
    """
    Samples points along the outer boundary of the ocean polygon.

    Args:
        ocean_df (GeoDataFrame): Ocean mask polygon layer.
        spacing (int): Distance between samples in coordinate space.
        sampling_method (int): Sampling strategy (currently supports 1 = uniform).

    Returns:
        List of shapely Point objects representing sampled coastline points.
    """
    methods = {
        1: "uniform"
    }

    if sampling_method not in methods:
        raise ValueError(f"Unsupported sampling_method {sampling_method}. Available: {list(methods.keys())}")

    ocean_geom = ocean_df.geometry.union_all()
    sampled_points = []

    for poly in ocean_df.geometry:
        for interior in poly.interiors:
            coords = list(interior.coords)
            for i in range(0, len(coords), int(spacing)):
                sampled_points.append(Point(coords[i]))

    return sampled_points

def draw_ray(start_point, angle, land_gdf, max_distance=2000, 
             proximity_thresh=5, dimensions=(1200, 1200)):
    """
    Cast a ray from start_point at given angle (degrees), clipped to land and bounding box.

    Returns:
        dict of ray metadata if valid, else None.
    """
    angle_rad = np.radians(angle)
    dx = np.cos(angle_rad) * max_distance
    dy = np.sin(angle_rad) * max_distance

    width, height = dimensions
    end_point = Point(start_point.x + dx, start_point.y + dy)

    # Determine hit_type based on boundary contact
    x, y = end_point.x, end_point.y
    if x <= 0 or y <= 0 or x >= width or y >= height:
        hit_type = "boundary"
    else:
        hit_type = "coastline"

    raw_ray = LineString([start_point, end_point])
        
    # Clip to bounding box
    bbox = box(0, 0, width, height)
    bounded_ray = raw_ray.intersection(bbox)
    if bounded_ray.is_empty or not isinstance(bounded_ray, LineString):
        return None

    # Clip to ocean (remove land portions)
    land_geom = land_gdf.geometry.union_all() if isinstance(land_gdf, gpd.GeoDataFrame) else land_gdf
    clipped = bounded_ray.difference(land_geom)

    segments = []
    if isinstance(clipped, LineString):
        segments = [clipped]
    elif hasattr(clipped, "geoms"):
        segments = [seg for seg in clipped.geoms if isinstance(seg, LineString)]

    for seg in segments:
        if seg.contains(start_point) or Point(seg.coords[0]).distance(start_point) <= proximity_thresh:
            return {
                'start': start_point,
                'end': Point(seg.coords[-1]),
                'angle': angle,
                'distance': start_point.distance(Point(seg.coords[-1])),
                'hit_type': hit_type,
                'valid': True,
                'geometry': seg
            }

    return None

def generate_rays_df(
    coastline_points,
    ocean_gdf,
    land_gdf,
    m=32,
    max_distance=2000,
    proximity_thresh=5,
    dimensions=(1200, 1200)
):
    """
    Generate a GeoDataFrame of valid rays from coastline points into ocean space.

    Parameters:
        coastline_points (list[Point]): Starting points for rays.
        ocean_gdf (GeoDataFrame): Ocean polygons (used for CRS and spatial context).
        land_gdf (GeoDataFrame): Land polygons to clip rays against.
        m (int): Number of rays per point (distributed 360° around).
        max_distance (float): Max length of any ray.
        proximity_thresh (float): Max distance from start point to valid ocean entry.

    Returns:
        GeoDataFrame: Valid rays as rows with geometry and metadata.
    """
    ray_records = []

    for i, point in enumerate(coastline_points):
        for j, angle in enumerate(np.linspace(0, 360, m, endpoint=False)):
            ray_info = draw_ray(
                start_point=point,
                angle=angle,
                land_gdf=land_gdf,
                max_distance=max_distance,
                proximity_thresh=proximity_thresh,
                dimensions=dimensions
            )

            if ray_info:
                ray_info["sample_index"] = i
                ray_info["angle_index"] = j
                ray_records.append(ray_info)

    return gpd.GeoDataFrame(ray_records, crs=ocean_gdf.crs)

if __name__ == '__main__':
    input_date = "042325"
    LAND_SHAPEFILE = f"land_{input_date}.shp"
    DRAW_FIG = True
    
    shapefile_path = directories.SHAPEFILES_DIR / LAND_SHAPEFILE
    # fig = world_viewer.plot_shapes(shapefile_path)
    # html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_input_landfiles.html"
    # viewing_util.save_figure_to_html(fig, html_path, open_on_export=True)
    land_gdf = union_shapefile([shapefile_path])

    image_width, image_height = get_image_dimensions(shapefiles.IMG_PATH)
    ocean_df = generate_voided_extent(shapefile_path, dimensions=(image_width, image_height))
    
    if DRAW_FIG:
        fig = world_viewer.plot_shapes(ocean_df)
        fig = world_viewer.plot_shapes(shapefile_path, fig)
    
    sampled_coastline_points = sample_perimeter_points(ocean_df)
    print('sampled coastline points:', len(sampled_coastline_points))
    
    if DRAW_FIG:
        fig = world_viewer.plot_overlay(fig, sampled_coastline_points, color="red", name="sample_points", size=5)
    
    print('generating rays')
    ray_df = generate_rays_df(sampled_coastline_points, ocean_df, land_gdf=land_gdf, m=32, 
                              max_distance=2000, proximity_thresh=5, dimensions=(image_width, image_height))
    print('ray_df:', ray_df.shape)
    print(ray_df[['start', 'end', 'geometry']].head())
    
    if DRAW_FIG:
        fig = world_viewer.plot_overlay(fig, ray_df, color="orange", name="rays", width=1)

    html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_ocean_mask.html"
    viewing_util.save_figure_to_html(fig, html_path, open_on_export=True)

    ray_df.to_file(directories.DATA_DIR / f"ocean_ray_dataset_{input_date}.geojson", driver="GeoJSON")

