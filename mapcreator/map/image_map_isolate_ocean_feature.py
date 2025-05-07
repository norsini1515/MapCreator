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
from mapcreator.scripts.extract_images import extract_image_from_file


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

def generate_ocean_mask(
    land_shapefile_paths: Union[Path, list[Path]],
    image_path: Path = None,
    default_dim: tuple = (1200, 1200)
) -> gpd.GeoDataFrame:
    """
    Generate a polygon layer representing the true ocean by subtracting land from the full image extent.

    Args:
        land_shapefile_paths (Path or list[Path]): One or more shapefile paths representing landmasses.
        image_path (Path, optional): Optional path to the image used in pipeline (used to infer size).
        default_dim (tuple): Default size (width, height) if image not provided.

    Returns:
        GeoDataFrame: GeoDataFrame representing the ocean mask polygon(s).
    """
    if isinstance(land_shapefile_paths, Path):
        land_shapefile_paths = [land_shapefile_paths]
    elif not isinstance(land_shapefile_paths, list):
        raise TypeError("land_shapefile_paths must be a Path or a list of Paths.")
    

    # Get image size or use default
    if image_path:
        img = extract_image_from_file(image_path)
        height, width = img.height, img.width
    else:
        width, height = default_dim

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

def sample_coastline(ocean_df: gpd.GeoDataFrame, spacing: int = 15, sampling_method: int = 1) -> list[Point]:
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
    # if ocean_geom.geom_type == "Polygon":
        # boundary = ocean_geom.exterior
        # num_points = int(boundary.length // spacing)
        # sampled_points = [boundary.interpolate(i * spacing) for i in range(num_points)]
    # elif ocean_geom.geom_type == "MultiPolygon":
    #     for poly in ocean_geom.geoms:
    #         boundary = poly.exterior
    #         num_points = int(boundary.length // spacing)
    #         sampled_points.extend([boundary.interpolate(i * spacing) for i in range(num_points)])
    # else:
    #     raise ValueError("Unsupported geometry type in ocean mask.")

    return sampled_points

def _extract_intersection_points(geometry_obj):
    """
    Helper to extract shapely Point geometries from a shapely or GeoSeries intersection.
    """
    if isinstance(geometry_obj, gpd.GeoSeries):
        geoms = geometry_obj.tolist()
    else:
        geoms = [geometry_obj]

    points = []
    for g in geoms:
        if g.is_empty:
            continue
        if isinstance(g, Point):
            points.append(g)
        elif hasattr(g, "geoms"):  # MultiPoint or GeometryCollection
            points.extend([x for x in g.geoms if isinstance(x, Point)])

    return points

def draw_ray(start_point, angle, ocean_gdf, land_gdf=None, max_distance=2000, proximity_thresh=5):
    """
    Casts a ray and returns metadata only if it enters ocean first, then hits land or boundary.

    Parameters:
        start_point (Point): Origin point.
        angle (float): Ray angle in radians.
        ocean_gdf (GeoDataFrame): Water polygons (single-row expected).
        land_gdf (GeoDataFrame, optional): Landmass polygons.
        max_distance (float): Maximum ray length.
        proximity_thresh (float): Max distance for ocean entry to count as valid.

    Returns:
        dict: Ray metadata with start, end, hit_type, and geometry if valid.
    """
    from shapely.geometry import LineString, Point
    from shapely import geometry

    # Build ray
    dx = np.cos(angle) * max_distance
    dy = np.sin(angle) * max_distance
    end_point = Point(start_point.x + dx, start_point.y + dy)
    ray = LineString([start_point, end_point])

    ocean_geom = ocean_gdf.geometry.iloc[0]
    ocean_boundary = ocean_geom.boundary

    # Get intersections
    ocean_hits = _extract_intersection_points(ray.intersection(ocean_geom))
    land_hits = _extract_intersection_points(land_gdf.intersection(ray)) if land_gdf is not None else []
    boundary_hits = _extract_intersection_points(ray.intersection(ocean_boundary))

    # Label and combine all hits
    hit_candidates = []
    hit_candidates += [(pt, "ocean") for pt in ocean_hits]
    hit_candidates += [(pt, "land") for pt in land_hits]
    hit_candidates += [(pt, "boundary") for pt in boundary_hits]

    # Sort by proximity to ray start
    hit_candidates.sort(key=lambda item: start_point.distance(item[0]))

    # Initialize output
    result = {
        'start_x': start_point.x,
        'start_y': start_point.y,
        'end_x': end_point.x,
        'end_y': end_point.y,
        'angle': np.degrees(angle),
        'distance': 0,
        'hit_type': None,
        'valid': False,
    }

    if not hit_candidates:
        return result

    first_point, first_type = hit_candidates[0]

    # Check that first hit is ocean and within threshold
    if first_type != "ocean" or start_point.distance(first_point) > proximity_thresh:
        return result

    # Now find the *next* feature hit (land or boundary) AFTER ocean
    for pt, hit_type in hit_candidates[1:]:
        if hit_type in ("land", "boundary"):
            result.update({
                'end_x': pt.x,
                'end_y': pt.y,
                'distance': start_point.distance(pt),
                'hit_type': hit_type,
                'valid': True,
            })
            break

    return result


def build_ray_dataset(ocean_gdf, land_gdf, sampled_points,
                      angle_step=15, max_distance=2000):
    ray_records = []

    test_index=4

    for i, point in enumerate(sampled_points):
        if test_index is not None and i >= test_index:
            break

        for angle_deg in range(0, 360, angle_step):
            angle_rad = np.radians(angle_deg)

            ray_info = draw_ray(point, angle_rad, ocean_gdf, land_gdf,
                                max_distance=max_distance)
            ray_info['sample_index'] = i  # Which sample point it came from
            
            # Build geometry
            ray_geom = LineString([(ray_info['start_x'], ray_info['start_y']),
                                   (ray_info['end_x'], ray_info['end_y'])])

            ray_info['geometry'] = ray_geom
            ray_records.append(ray_info)


    # Build GeoDataFrame
    ray_gdf = gpd.GeoDataFrame(ray_records, crs="EPSG:4326")  # or None if purely pixel space

    return ray_gdf

def generate_rays_df(coastline_points, ocean_gdf, land_gdf=None, m=32, max_distance=2000, proximity_thresh=5):
    """
    Generate a GeoDataFrame of valid rays from coastline points into ocean space.

    Parameters:
        coastline_points (list[Point]): Starting points along the coastline.
        ocean_gdf (GeoDataFrame): Ocean geometry (should be single-row).
        land_gdf (GeoDataFrame, optional): Land geometry for detecting land hits.
        m (int): Number of rays per point.
        max_distance (float): Maximum ray length.
        proximity_thresh (float): Max distance from start to initial ocean hit.

    Returns:
        GeoDataFrame: Rays with metadata and geometry.
    """
    ray_records = []

    for i, point in enumerate(coastline_points):
        if(i == 15):
            print('enough for now')
            break
        for j, angle in enumerate(np.linspace(0, 2 * np.pi, m, endpoint=False)):
            ray_info = draw_ray(
                start_point=point,
                angle=angle,
                ocean_gdf=ocean_gdf,
                land_gdf=land_gdf,
                max_distance=max_distance,
                proximity_thresh=proximity_thresh
            )
            # print(i, angle, ray_info["valid"])
            if ray_info["valid"]:
                ray_info["start"] = geometry.Point(ray_info["start_x"], ray_info["start_y"])
                ray_info["end"] = geometry.Point(ray_info["end_x"], ray_info["end_y"])
                ray_info["sample_index"] = i
                ray_info["angle_index"] = j
                ray_info["geometry"] = LineString([ray_info["start"], ray_info["end"]])
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
    ocean_df = generate_ocean_mask(shapefile_path, shapefiles.IMG_PATH)
    # print('land_df:', type(land_gdf), "\nocean_df: ", type(ocean_df))
    # sys.exit()
    # print(ocean_df.head(5))
    # shapefiles.visualize_shapefile(ocean_df)
    # exit()
    if DRAW_FIG:
        fig = world_viewer.plot_shapes(ocean_df)
        fig = world_viewer.plot_shapes(shapefile_path, fig)
    
    sampled_points = sample_coastline(ocean_df)
    print('sampled points:', len(sampled_points))
    
    if DRAW_FIG:
        fig = world_viewer.plot_overlay(fig, sampled_points, color="red", name="sample_points", size=5)
    
    # ray_df = build_ray_dataset(ocean_df, land_gdf, sampled_points)
    ray_df = generate_rays_df(sampled_points, ocean_df, land_gdf=land_gdf, m=32, max_distance=2000, proximity_thresh=5)
    print('ray_df:', ray_df.shape)
    print(ray_df[['start', 'end', 'geometry']].head())
    
    fig = world_viewer.plot_overlay(fig, ray_df, color="orange", name="rays", width=1)

    html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_ocean_mask.html"
    viewing_util.save_figure_to_html(fig, html_path, open_on_export=False)

