'''
Workflow for Contour Extraction
Load the Processed Image:

Use OpenCV to load the binarized image or a NumPy array representation of your preprocessed image.
Extract Contours:

Use OpenCV's cv2.findContours function to detect the contours of the landmass or other features.
Convert Contours to Polygons:

Convert the extracted contours into geometries compatible with shapefile formats (e.g., using Shapely).
Save to Shapefile:

Use Fiona or pyshp to save the contour polygons into a shapefile for use in GIS software.
'''
import os
import cv2
import fiona
import numpy as np
import pandas as pd
import geopandas as gpd
from fiona.crs import CRS
from datetime import datetime
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
from pathlib import Path

from mapcreator import directories, configs, preprocess_image
from mapcreator.scripts import extract_images
from mapcreator import world_viewer
from mapcreator.visualization import viewing_util

VERBOSE = False

def extract_contours(img_array, min_area=configs.MIN_AREA, min_points=configs.MIN_POINTS,
                     verbose=VERBOSE):
    """
    Extract valid contours from a binary image.

    Args:
        img_array: NumPy array representing a binary image.
        min_area: Minimum area threshold (default = 5.0 pixels).
        min_points: Minimum number of points in a valid polygon.

    Returns:
        List of valid Shapely Polygons.
    """
    contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    
    if verbose:
        print(len(contours), "potential polygons found")
    for i, c in enumerate(contours):
        if len(c) >= min_points:  # Ensure enough points for a polygon
            coords = [(x, y) for x, y in c[:, 0, :]]

            polygon = Polygon(coords)

            # Apply enhanced filtering
            if polygon.is_valid and polygon.area >= min_area and len(polygon.exterior.coords) >= min_points:
                if polygon.length > min_area * 0.5:  # Avoid extremely thin polygons
                    polygons.append(polygon)
                else:
                    if verbose:
                        print(f"⚠️ Skipping THIN Polygon {i} | Area: {polygon.area:.2f} | Length: {polygon.length:.2f}")
            else:
                if verbose:
                    reason = explain_validity(polygon)
                    print(f"❌ Skipping Invalid Polygon {i} | Area: {polygon.area:.2f} | Points: {len(polygon.exterior.coords)}")
                    print(f"\tReason: {reason}")
                
                fixed = polygon.buffer(0)

                # Handle Polygon or MultiPolygon
                if isinstance(fixed, Polygon):
                    candidates = [fixed]
                elif isinstance(fixed, MultiPolygon):
                    candidates = list(fixed.geoms)
                else:
                    candidates = []
                
                if verbose:    
                    print(f'{len(candidates)} candidates for polygon {i}.')
                
                for j, poly in enumerate(candidates):
                    if poly.is_valid and poly.area >= min_area and len(poly.exterior.coords) >= min_points:
                        polygons.append(poly)
                        if verbose:
                            print(f"✅ Polygon {i}.{j} repaired using buffer(0) and added.")
                    else:
                        if verbose:
                            print(f"⚠️ Polygon {i}.{j} from repair was skipped (invalid or too small).")
            if verbose:
                print()
            
    return polygons

    print(f"✅ Shapefile saved to: {shapefile_path}")
   
def visualize_shapefile(data, title: str = None):
    """
    Visualizes a shapefile or in-memory polygons using GeoPandas and Matplotlib.

    Args:
        data (Path | GeoDataFrame | List[Polygon]): 
            Path to a shapefile, a GeoDataFrame, or a list of Shapely polygons.
        title (str): Optional plot title (defaults to filename or "Polygon Visualization").
    """
    if isinstance(data, (str, Path)):
        gdf = gpd.read_file(data)
        source = Path(data).stem
    elif isinstance(data, gpd.GeoDataFrame):
        gdf = data
        source = title or "GeoDataFrame"
    elif isinstance(data, list):
        gdf = gpd.GeoDataFrame({"geometry": data}, crs="EPSG:4326")
        source = title or "Polygon List"
    else:
        raise TypeError("Input must be a Path, GeoDataFrame, or list of Shapely Polygons.")

    if gdf.empty:
        raise ValueError("GeoDataFrame is empty, cannot visualize.")
    
    title = title or f"Visualization of {source}"
    print(f"🗺️ Displaying: {source}")
    print("Bounds:", gdf.total_bounds)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    if np.isinf(xmin) or np.isinf(ymin) or np.isinf(xmax) or np.isinf(ymax):
        raise ValueError("Shapefile contains invalid coordinate bounds.")
    
    fig, ax = plt.subplots(figsize=(10, 10), num=title)
    gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue", aspect=None)
    
    #set aspect accordingly
    width = xmax - xmin
    height = ymax - ymin
    ax.set_aspect(width / height if width > 0 and height > 0 else "equal", adjustable="datalim")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    bbox = plt.Rectangle((xmin, ymin), width, height,
                         linewidth=1, edgecolor='gray', facecolor='none', linestyle=':')
    ax.add_patch(bbox)

    ax.invert_yaxis()
    plt.title(title)
    plt.show()
       
def build_geodataframe(polygons, metadata=None):
    """
    Construct a GeoDataFrame from a list of polygons and optional metadata.

    Args:
        polygons (List[Polygon]): Shapely polygons.
        metadata (dict): Metadata to apply to all polygons.

    Returns:
        GeoDataFrame: A GeoPandas dataframe with metadata and geometry.
    """
    metadata = metadata or {}
    df = pd.DataFrame(metadata, index=range(len(polygons)))
    gdf = gpd.GeoDataFrame(df, geometry=polygons, crs="EPSG:4326")
    gdf["id"] = gdf.index  # Always include an ID
    
    return gdf

def image_to_geometry_pipeline(
    img_path: Path,
    metadata: dict = None,
    visualize: bool = True, 
    contrast_factor: float = 2.0, 
    min_area: float = 1.0, 
    min_points: int = 3, 
    ):
    """
    End-to-end pipeline to convert an image to a shapefile of extracted polygons.

    Args:
        img_path (Path): Path to the input .jpg image.
        invert (bool): Whether to invert image after binarization.
        output (bool): If True, save the shapefile.
        shapefile_path (Path): Where to save the shapefile if output=True.
        visualize (bool): Whether to visualize the shapefile.
        contrast_factor (float): Contrast enhancement factor.
        min_area (float): Minimum polygon area to keep.
        min_points (int): Minimum number of points per polygon.
        metadata (dict): Metadata dictionary to apply to all features.

    Returns:
        GeoDataFrame
    """
     # --- Step 1: Preprocess Image ---
    print("Preprocessing image")
    img_array = preprocess_image(
        img_path,
        contrast_factor=contrast_factor,
        invert=metadata.get("invert", False),
        flood_fill=metadata.get("flood_fill", False),
    )
    # --- Step 2: Extract Polygons ---
    print("Extracting contours...")
    polygons = extract_contours(img_array, min_area=min_area, min_points=min_points)
    print(f"Detected {len(polygons)} polygons.")

    # --- Step 3: Wrap into GeoDataFrame ---
    print("Building geodataframe and attaching metadata.")
    gdf = build_geodataframe(polygons, metadata)

    if visualize:
        visualize_shapefile(gdf, title=img_path.name)

    return gdf

def export_geometry(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    driver: str = None,
):
    """
    Export a GeoDataFrame to GeoJSON or Shapefile based on file extension.

    Args:
        gdf (GeoDataFrame): The data to export.
        output_path (Path): File path ending in .geojson or .shp.
        driver (str): Optional override for driver (e.g., "GeoJSON", "ESRI Shapefile").

    Raises:
        ValueError: If file extension is unsupported and no driver is specified.
    """
    ext = output_path.suffix.lower()

    if not driver:
        if ext == ".geojson":
            driver = "GeoJSON"
        elif ext == ".shp":
            driver = "ESRI Shapefile"
        else:
            raise ValueError(f"Unsupported extension '{ext}'. Specify a driver explicitly.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver=driver)

if __name__ == '__main__':
    #CONSTANTS
    DATE = datetime.now().strftime('%m%d%y')
    OUTPUT=True
    VISUALIZE=False
    USER_DEBUG_MODE=False
    
    IMG_PATH = directories.IMAGES_DIR / configs.WORKING_WORLD_IMG_NAME
    #---------
    #image variables       
    extract_images.display_image(IMG_PATH, title=f'Original Image, {configs.WORKING_WORLD_IMG_NAME}')
    #---------
    
    combined_fig = None #combined figure of all shapefiles being processed
    #---------
    
    print('processing base land geometry')
    land_meta = configs.GEOMETRY_METADATA["land"].copy()
    land_img = directories.IMAGES_DIR / land_meta['source']
    land_geometry_path = directories.SHAPEFILES_DIR / f"land_{DATE}.geojson"
    land_gdf = image_to_geometry_pipeline(land_img, visualize=VISUALIZE, metadata=land_meta)
    export_geometry(land_gdf, land_geometry_path)
    
    print('processing internal water geometry')
    lakes_meta = configs.GEOMETRY_METADATA["lakes"].copy()
    lakes_img = directories.IMAGES_DIR / land_meta['source']
    lakes_geometry_path = directories.SHAPEFILES_DIR / f"lakes_{DATE}.geojson"
    lakes_gdf = image_to_geometry_pipeline(lakes_img, visualize=VISUALIZE, metadata=lakes_meta)
    export_geometry(lakes_gdf, lakes_geometry_path)
    
    combined_fig = world_viewer.plot_shapes(land_gdf)
    combined_fig = world_viewer.plot_shapes(lakes_gdf, combined_fig)
    combined_html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_base_geometries_{DATE}.html"
    viewing_util.save_figure_to_html(combined_fig, combined_html_path, open_on_export=True)