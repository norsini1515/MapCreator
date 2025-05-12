"""
📡 cast_rays.py

Purpose:
--------
Cast directional rays from the perimeter of a voided extent (e.g., ocean mask) to analyze
spatial openness relative to holes (e.g., landmasses). Rays help identify enclosed or 
connected water features like fjords or bays.

Main Steps:
-----------
1. Load extent and hole geometries.
2. Sample perimeter points along interior boundaries.
3. Cast rays at evenly spaced angles.
4. Clip rays against land and bounding box.
5. Classify rays as 'boundary' or 'coastline'.
6. Save results to GeoJSON and optionally visualize.

Features:
---------
- Parallel ray casting using `ProcessPoolExecutor`
- Modular sampling and ray generation
- Produces a ray GeoDataFrame with angle, distance, hit_type, and geometry
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box, LineString
from mapcreator import directories, configs, world_viewer, summarize_load
from mapcreator.map import export_geometry, union_geo_files
from mapcreator.visualization import viewing_util
from mapcreator.scripts.extract_images import get_image_dimensions
from pathlib import Path
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile

def sample_perimeter_points(extent_gdf: gpd.GeoDataFrame, spacing: int = 15, sampling_method: int = 1) -> list[Point]:
    """
    Samples points along the outer boundary of the ocean polygon.

    Args:
        extent_gdf (GeoDataFrame): Ocean mask polygon layer.
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

    ocean_geom = extent_gdf.geometry.union_all()
    sampled_points = []

    for poly in extent_gdf.geometry:
        for interior in poly.interiors:
            coords = list(interior.coords)
            for i in range(0, len(coords), int(spacing)):
                sampled_points.append(Point(coords[i]))

    return sampled_points

def draw_ray(start_point, angle, holes_gdf, max_distance=2000, 
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

    raw_ray = LineString([start_point, end_point])
        
    # Clip to bounding box
    bbox = box(0, 0, width, height)
    bounded_ray = raw_ray.intersection(bbox)
    if bounded_ray.is_empty or not isinstance(bounded_ray, LineString):
        return None

    # Clip to ocean (remove land portions)
    holes_geom = holes_gdf.geometry.union_all() if isinstance(holes_gdf, gpd.GeoDataFrame) else holes_gdf
    clipped = bounded_ray.difference(holes_geom)

    segments = []
    if isinstance(clipped, LineString):
        segments = [clipped]
    elif hasattr(clipped, "geoms"):
        segments = [seg for seg in clipped.geoms if isinstance(seg, LineString)]

    for seg in segments:
        if seg.contains(start_point) or Point(seg.coords[0]).distance(start_point) <= proximity_thresh:
            final_end = Point(seg.coords[-1])

            # Hit type decision based on actual final endpoint
            x, y = final_end.x, final_end.y

            if x <= 0 or y <= 0 or x >= width or y >= height:
                hit_type = "boundary"
            else:
                hit_type = "coastline"
            
            return {
                'angle': angle,
                'distance': start_point.distance(final_end),
                'hit_type': hit_type,
                'geometry': seg
            }

    return None

def cast_rays_from_point(args):
    point, holes_gdf_serialized, ray_count, max_distance, proximity_thresh, dimensions = args
    holes_gdf = gpd.read_file(holes_gdf_serialized)  # reload to avoid pickle issues

    rays = []
    for angle in np.linspace(0, 360, ray_count, endpoint=False):
        ray_info = draw_ray(
            start_point=point,
            angle=angle,
            holes_gdf=holes_gdf,
            max_distance=max_distance,
            proximity_thresh=proximity_thresh,
            dimensions=dimensions
        )
        if ray_info:
            rays.append(ray_info)

    return rays

def generate_rays_df_parallel(
    sampled_points,
    holes_gdf,
    ray_count=32,
    max_distance=2000,
    proximity_thresh=5,
    dimensions=(1200, 1200),
    max_workers=8
):
    ray_records = []

    tmp = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
    holes_gdf_path = Path(tmp.name)
    tmp.close()  # Ensure the file is closed before writing to it
    holes_gdf.to_file(holes_gdf_path, driver="GeoJSON")


    args_list = [
        (pt, holes_gdf_path, ray_count, max_distance, proximity_thresh, dimensions)
        for pt in sampled_points
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(cast_rays_from_point, args) for args in args_list]
        for future in as_completed(futures):
            result = future.result()
            ray_records.extend(result)

    try:
        holes_gdf_path.unlink()
    except Exception as e:
        print(f"⚠️ Warning: Could not delete temp file {holes_gdf_path}: {e}")

    return gpd.GeoDataFrame(ray_records, crs="EPSG:4326")

#---------------------------------

DRAW_FIG = True
if __name__ == '__main__':
    #setting variables
    input_date = "050725"
    output_name = 'rays_cast'
    output_date = '051025'
    #---------
    #define land (holes)
    LAND_GEOJSON = f"land_{input_date}.geojson"
    land_path = directories.SHAPEFILES_DIR / LAND_GEOJSON
    #---------
    #define ocean (extent)
    OCEAN_GEOJSON = f"ocean_{input_date}.geojson"
    ocean_path = directories.SHAPEFILES_DIR / OCEAN_GEOJSON
    #---------
    #load holes and extent 
    land_gdf = union_geo_files([land_path])
    ocean_gdf = gpd.read_file(ocean_path)
    #sumarize loads of holes and extent
    summarize_load(land_gdf, 'land_gdf')
    summarize_load(ocean_gdf, 'ocean_gdf')
    #---------
    print("sample outlines of the holes (the edges of the extent file)")
    sampled_coastline_points = sample_perimeter_points(ocean_gdf)
    print('sampled coastline points:', len(sampled_coastline_points))
    #---------
    print("visually sumarize preprocessing for ray casting")
    if DRAW_FIG:
        fig = world_viewer.plot_shapes(ocean_gdf)
        fig = world_viewer.plot_shapes(land_gdf, fig)
        fig = world_viewer.plot_overlay(fig, sampled_coastline_points, color="red", name="sample_points", size=5)
    #---------
    print('getting source image dimensions')
    IMG_PATH = directories.IMAGES_DIR / configs.WORKING_WORLD_IMG_NAME
    image_width, image_height = get_image_dimensions(IMG_PATH)
    
    print('generating rays . . .')
    ray_df = generate_rays_df_parallel(sampled_points=sampled_coastline_points, 
                                       holes_gdf=land_gdf, ray_count=32, 
                                       max_distance=2000, proximity_thresh=5, 
                                       dimensions=(image_width, image_height),
                                       max_workers=8)
    print('ray_df:', ray_df.shape)

    print('outputing ray data set')
    rays_filename = f"ocean_ray_dataset_{output_date}.geojson"
    rays_filepath = directories.DATA_DIR / rays_filename
    export_geometry(ray_df, rays_filepath)
    
    print(ray_df[['angle', 'distance', 'hit_type', 'geometry']].head())
    
    print('display results')
    if DRAW_FIG:
        fig = world_viewer.plot_overlay(fig, ray_df, color="orange", name="rays", width=1)
        html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_rays_cast.html"
        viewing_util.save_figure_to_html(fig, html_path, open_on_export=True)