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
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Union
from shapely.geometry import Polygon, MultiPolygon, Point, box

from mapcreator import world_viewer
from mapcreator import directories, configs
import mapcreator.map.shapefile as shapefiles
from mapcreator.visualization import viewing_util
from mapcreator.scripts.extract_images import extract_image_from_file


def union_shapefile(land_shapefile_paths):
    """
    Union all geometries from one or more shapefiles into a single geometry.

    Args:
        land_shapefile_paths (list[Path]): List of shapefile paths.

    Returns:
        Shapely geometry: Unified geometry of all land features.
    """
    all_land_geoms = []
    for path in land_shapefile_paths:
        gdf = gpd.read_file(path)
        all_land_geoms.append(gdf.geometry)  # append GeoSeries, not just .geometry values

    # Concatenate into one GeoSeries and apply union_all
    all_geoms_series = gpd.GeoSeries(pd.concat(all_land_geoms, ignore_index=True))
    return all_geoms_series.union_all()


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
    
    land_union = union_shapefile(land_shapefile_paths)
    
    # Get image size or use default
    if image_path:
        img = extract_image_from_file(image_path)
        height, width = img.height, img.width
    else:
        width, height = default_dim

    # Create outer bounding box representing the full image extent
    world_bbox = box(0, 0, width, height)

    # Subtract land polygons from the bounding box to isolate the ocean
    ocean_geom = world_bbox.difference(land_union)
    
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

if __name__ == '__main__':
    input_date = "042325"
    LAND_SHAPEFILE = f"land_{input_date}.shp"
    
    shapefile_path = directories.SHAPEFILES_DIR / LAND_SHAPEFILE
    # fig = world_viewer.plot_shapes(shapefile_path)
    # html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_input_landfiles.html"
    # viewing_util.save_figure_to_html(fig, html_path, open_on_export=True)

    ocean_df = generate_ocean_mask(shapefile_path, shapefiles.IMG_PATH)
    print(ocean_df.head(5))
    # shapefiles.visualize_shapefile(ocean_df)
    # exit()
    
    fig = world_viewer.plot_shapes(ocean_df)
    fig = world_viewer.plot_shapes(shapefile_path, fig)
    
    sampled_points = sample_coastline(ocean_df)
    print('sampled points:', len(sampled_points))
    fig = world_viewer.plot_overlay(fig, sampled_points, color="red", name="sample_points", size=3)

    html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_ocean_mask.html"
    viewing_util.save_figure_to_html(fig, html_path, open_on_export=False)

