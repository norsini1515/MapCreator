"""
🌊 geometry_mask_builder.py

Purpose:
--------
Construct a voided extent (e.g., ocean mask) by subtracting hole geometries 
(e.g., landmasses) from a full map bounding box.

Key Functions:
--------------
- union_geo_files: Combines multiple shapefiles or GeoDataFrames into one geometry.
- subtract_geometries_from_extent: Builds the masked region by removing holes from a rectangular extent.

Typical Use Case:
-----------------
Used to define spatial extents (like oceans) before downstream analysis such as 
ray casting, pinch detection, or region segmentation.

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

def union_geo_files(
    sources: Union[Path, gpd.GeoDataFrame, list[Union[Path, gpd.GeoDataFrame]]]
) -> gpd.GeoDataFrame:
    """
    Union all geometries from one or more shapefiles or GeoDataFrames into a single-row GeoDataFrame.
    Args:
        sources (Path | GeoDataFrame | list): Single or list of shapefile paths or GeoDataFrames.
    Returns:
        GeoDataFrame: One-row GeoDataFrame containing the unioned geometry.
    """
    if not isinstance(sources, list):
        sources = [sources]

    geoms = []
    for item in sources:
        if isinstance(item, (str, Path)):
            gdf = gpd.read_file(item)
            geoms.append(gdf.geometry)
        elif isinstance(item, gpd.GeoDataFrame):
            geoms.append(item.geometry)
        else:
            raise TypeError(f"Unsupported input type: {type(item)}")

    all_geoms = gpd.GeoSeries(pd.concat(geoms, ignore_index=True))
    unioned = all_geoms.union_all()

    return gpd.GeoDataFrame({'geometry': [unioned]}, crs="EPSG:4326")

def subtract_geometries_from_extent(
        geo_files: Union[Path, gpd.GeoDataFrame, list[Union[Path, gpd.GeoDataFrame]]],
        dimensions: tuple = (1200, 1200)) -> gpd.GeoDataFrame:
    """
    Subtracts input geometries from a bounding box to create a voided extent mask.

    Args:
        geo_file (Path | GeoDataFrame | list): Shapefile path(s) or GeoDataFrame(s) defining the "hole" regions.
        dimensions (tuple): (width, height) of the bounding extent.

    Returns:
        GeoDataFrame: A single-row GeoDataFrame representing the voided geometry.
    """
    
    # Subtract polygons from the bounding box
    unioned_geoms = union_geo_files(geo_files).geometry.iloc[0]
    
    # Create outer bounding box representing the full image extent
    bbox = box(0, 0, *dimensions)
    voided = bbox.difference(unioned_geoms)
    voided_geom = gpd.GeoDataFrame({'geometry': [voided]}, crs="EPSG:4326")
    
    #------
    print("Ocean Geometry Type:", type(voided_geom))
    if isinstance(voided_geom, Polygon):
        print(" - Exterior length:", len(voided_geom.exterior.coords))
        print(" - Number of holes:", len(voided_geom.interiors))
    elif isinstance(voided_geom, MultiPolygon):
        for i, poly in enumerate(voided_geom.geoms):
            print(f" - Polygon {i}: {len(poly.exterior.coords)} exterior coords, {len(poly.interiors)} holes")
    #------

    voided_gdf = gpd.GeoDataFrame({'geometry': [voided]}, crs="EPSG:4326")

    return voided_gdf
    
if __name__ == '__main__':
    input_date = "050725"
    LAND_GEOJSON = f"land_{input_date}.geojson"
    DRAW_FIG = True
    
    geo_file_path = directories.SHAPEFILES_DIR / LAND_GEOJSON
    land_gdf = union_geo_files([geo_file_path])

    #get source image dimensions
    IMG_PATH = directories.IMAGES_DIR / configs.WORKING_WORLD_IMG_NAME
    image_width, image_height = get_image_dimensions(IMG_PATH)

    ocean_gdf = subtract_geometries_from_extent(geo_file_path, dimensions=(image_width, image_height))
    ocean_gdf["type"] = "ocean"
    ocean_gdf["source"] = ";".join(p for p in [geo_file_path.name])

    if DRAW_FIG:
        fig = world_viewer.plot_shapes(ocean_gdf)
        fig = world_viewer.plot_shapes(land_gdf, fig)
    
    ocean_path = directories.SHAPEFILES_DIR / f"ocean_{input_date}.geojson"
    export_geometry(ocean_gdf, ocean_path)
    
    html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_ocean_mask.html"
    viewing_util.save_figure_to_html(fig, html_path, open_on_export=True)
