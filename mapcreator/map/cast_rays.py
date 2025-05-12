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

def generate_ray_dataset(
    holes_path: Path,
    extent_path: Path,
    img_path: Path,
    output_path = Path,
    ray_count: int = 32,
    spacing: int = 15,
    max_distance: float = 2000.0,
    proximity_thresh: int = 5,
    max_workers: int = 8,
    draw_fig: bool = True,
    export: bool = True
) -> gpd.GeoDataFrame:
    """
    Pipeline function to generate and optionally export a ray dataset.

    Args:
        holes_path: Path to landmass (hole) GeoJSON or Shapefile.
        extent_path: Path to void extent (e.g., ocean mask).
        img_path: Path to source image file (used for dimension clipping).
        output_path: File path to save ray dataset (should end in .geojson or .shp).
        ray_count: Number of rays to cast per sampled point.
        spacing: Distance between sampled points on perimeter.
        max_distance: Maximum length of a ray.
        proximity_thresh: Distance threshold to determine valid segment.
        max_workers: Number of processes to use.
        draw_fig: Whether to visualize the geometries and rays.
        export: Whether to save the ray dataset to disk.

    Returns:
        GeoDataFrame of valid rays with metadata.
    """
    # --- Load geometries ---
    print("load files . . .")
    holes_gdf = union_geo_files([holes_path])
    extent_gdf = gpd.read_file(extent_path)
    summarize_load(holes_gdf, "holes_gdf")
    summarize_load(extent_gdf, "extent_gdf")

    # --- Sample Points ---
    print('sampling border')
    sampled_points = sample_perimeter_points(extent_gdf, spacing=spacing)
    print(f"Sampled {len(sampled_points)} perimeter points.")

    # --- Dimensions ---
    dimensions = get_image_dimensions(img_path)

    # --- Visualization Prep ---
    if draw_fig:
        fig = world_viewer.plot_shapes(extent_gdf)
        fig = world_viewer.plot_shapes(holes_gdf, fig)
        fig = world_viewer.plot_overlay(fig, sampled_points, color="red", name="sample_points", size=5)

    # --- Ray Casting ---
    print("Casting rays...")
    ray_df = generate_rays_df_parallel(
        sampled_points=sampled_points,
        holes_gdf=holes_gdf,
        ray_count=ray_count,
        max_distance=max_distance,
        proximity_thresh=proximity_thresh,
        dimensions=dimensions,
        max_workers=max_workers
    )
    print(f"Generated {len(ray_df)} rays.")

    # --- Export ---
    if export:
        export_geometry(ray_df, output_path)
        print(f"Exported rays to {output_path}")

    if draw_fig:
        fig = world_viewer.plot_overlay(fig, ray_df, color="orange", name="rays", width=1)
        viewing_util.save_figure_to_html(fig, output_path.with_suffix('.html'), open_on_export=True)

    return ray_df

#---------------------------------

if __name__ == '__main__':
    DRAW_FIG = True
    EXPORT = True

    land_path = directories.SHAPEFILES_DIR / "land_050725.geojson"
    ocean_path = directories.SHAPEFILES_DIR / "ocean_050725.geojson"
    img_path = directories.IMAGES_DIR / configs.WORKING_WORLD_IMG_NAME
    output_path = directories.DATA_DIR / "ocean_ray_dataset_051025.geojson"

    generate_ray_dataset(
        holes_path=land_path,
        extent_path=ocean_path,
        img_path=img_path,
        output_path=output_path,
        ray_count=32,
        spacing=15,
        max_distance=2000,
        proximity_thresh=5,
        max_workers=8,
        draw_fig=DRAW_FIG,
        export=EXPORT
    )
