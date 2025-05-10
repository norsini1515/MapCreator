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
from shapely.geometry import Polygon, MultiPolygon, box

from mapcreator import directories, configs, world_viewer
from mapcreator.map import export_geometry
from mapcreator.visualization import viewing_util
from mapcreator.scripts.extract_images import get_image_dimensions

def union_geo_files(land_shapefile_paths):
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

def generate_voided_extent(geo_file_paths: Union[Path, list[Path]],
                            dimensions: tuple = (1200, 1200)) -> gpd.GeoDataFrame:
    """
    Generate a polygon layer representing the true ocean by subtracting land from the full image extent.

    Args:
        land_shapefile_paths (Path or list[Path]): One or more shapefile paths representing landmasses.
        default_dim (tuple): Default size (width, height) if image not provided.

    Returns:
        GeoDataFrame: GeoDataFrame representing the ocean mask polygon(s).
    """
    if isinstance(geo_file_paths, Path):
        geo_file_paths = [geo_file_paths]
    elif not isinstance(geo_file_paths, list):
        raise TypeError("land_shapefile_paths must be a Path or a list of Paths.")

    # Get image size or use default
    width, height = dimensions

    # Create outer bounding box representing the full image extent
    world_bbox = box(0, 0, width, height)

    # Subtract land polygons from the bounding box to isolate the ocean
    land_geom  = union_geo_files(geo_file_paths).geometry.iloc[0]
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
    ocean_gdf["source"] = ";".join(p.name for p in geo_file_paths)

    return ocean_gdf

if __name__ == '__main__':
    input_date = "050725"
    LAND_GEOJSON = f"land_{input_date}.geojson"
    DRAW_FIG = True
    
    geo_file_path = directories.SHAPEFILES_DIR / LAND_GEOJSON
    land_gdf = union_geo_files([geo_file_path])
    

    IMG_PATH = directories.IMAGES_DIR / configs.WORKING_WORLD_IMG_NAME
    image_width, image_height = get_image_dimensions(IMG_PATH)
    ocean_gdf = generate_voided_extent(geo_file_path, dimensions=(image_width, image_height))

    if DRAW_FIG:
        fig = world_viewer.plot_shapes(ocean_gdf)
        fig = world_viewer.plot_shapes(land_gdf, fig)
    
    ocean_path = directories.SHAPEFILES_DIR / f"ocean_{input_date}.geojson"
    export_geometry(ocean_gdf, ocean_path)
    
    html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_ocean_mask.html"
    viewing_util.save_figure_to_html(fig, html_path, open_on_export=True)
