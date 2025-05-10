import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box, LineString
from mapcreator import directories, configs, world_viewer
from mapcreator.map import export_geometry, union_geo_files
from mapcreator.visualization import viewing_util
from mapcreator.scripts.extract_images import get_image_dimensions
import sys


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
    DRAW_FIG = True
    input_date = "050725"

    LAND_GEOJSON = f"land_{input_date}.geojson"
    land_path = directories.SHAPEFILES_DIR / LAND_GEOJSON
    
    OCEAN_GEOJSON = f"ocean_{input_date}.geojson"
    ocean_path = directories.SHAPEFILES_DIR / OCEAN_GEOJSON

    land_gdf = union_geo_files([land_path])
    ocean_gdf = gpd.read_file(ocean_path)
    
    print('land_gdf', land_gdf.shape, f'\ncolumns: {land_gdf.columns.tolist()}')
    print()
    print('ocean_gdf', ocean_gdf.shape, f'\ncolumns: {ocean_gdf.columns.tolist()}')


    sampled_coastline_points = sample_perimeter_points(ocean_gdf)
    print('sampled coastline points:', len(sampled_coastline_points))
    
    if DRAW_FIG:
        fig = world_viewer.plot_shapes(ocean_gdf)
        fig = world_viewer.plot_shapes(land_gdf, fig)
        # fig = world_viewer.plot_overlay(fig, sampled_coastline_points, color="red", name="sample_points", size=5)

    IMG_PATH = directories.IMAGES_DIR / configs.WORKING_WORLD_IMG_NAME
    image_width, image_height = get_image_dimensions(IMG_PATH)
    
    print('generating rays . . .')
    ray_df = generate_rays_df(sampled_coastline_points, ocean_gdf, land_gdf=land_gdf, m=32, 
                              max_distance=2000, proximity_thresh=5, dimensions=(image_width, image_height))
    print('ray_df:', ray_df.shape)
    print(ray_df[['start', 'end', 'geometry']].head())
    
    if DRAW_FIG:
        fig = world_viewer.plot_overlay(fig, ray_df, color="orange", name="rays", width=1)

    if DRAW_FIG:
        html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_test_output.html"
        viewing_util.save_figure_to_html(fig, html_path, open_on_export=True)

    ray_df.to_file(directories.DATA_DIR / f"ocean_ray_dataset_{input_date}.geojson", driver="GeoJSON")