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

from mapcreator.map.image_map_pre_edits import flood_fill_ocean
from mapcreator import directories, configs
import mapcreator.scripts.extract_images as extract_images
from mapcreator import polygon_viewer, world_viewer
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

def image_to_shapefile(
    img_path: Path,
    invert: bool = False,
    output: bool = False,
    shapefile_path: Path = None,
    visualize: bool = True, 
    contrast_factor: float = 2.0, 
    min_area: float = 1.0, 
    min_points: int = 3, 
    metadata: dict = None,
    flood_fill: bool=False,
    verbose: bool=VERBOSE
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
    if not str(img_path).endswith(".jpg"):
        raise ValueError("Expecting a .jpg input image.")

    if output and shapefile_path is None:
        raise ValueError("Output path must be specified if output=True.")
        
    img = extract_images.extract_image_from_file(img_path)
    img_array = extract_images.prepare_image(img, contrast_factor=contrast_factor)
    
    if invert:
        img_array = extract_images.invert_image(img_array)
        
    if flood_fill:
        extract_images.display_image(img_array, title='Displaying BEFORE Floodfill')
        img_array = flood_fill_ocean(img_array)
        extract_images.display_image(img_array, title='Displaying AFTER Floodfill')
        
    print("Extracting contours...")
    polygons = extract_contours(img_array, min_area, min_points, verbose)
    print(f"✅ Detected {len(polygons)} polygons.")
    if visualize:
        visualize_shapefile(polygons, title=shapefile_path.stem)
    
    print("Building geodataframe and attaching metadata.")
    gdf = build_geodataframe(polygons, metadata)
    #Use plotly plot to have hover display metadata.
    # if visualize:
    #     visualize_shapefile(gdf)
        
    if output:
        gdf.to_file(shapefile_path)
        print(f"✅ Shapefile written to: {shapefile_path}")
    
    return gdf

if __name__ == '__main__':
    #CONSTANTS
    DATE = datetime.now().strftime('%m%d%y')
    OUTPUT=False
    VISUALIZE=True
    USER_DEBUG_MODE=False
    
    # image_name = "SauceField2.jpg"
    # img_path = directories.IMAGES_DIR / image_name
    # extract_images.display_image(img_path, title=f'Original Image, {image_name}')
    
    
    # shapefile_path = directories.SHAPEFILES_DIR / f"{image_name[:-3]}.shp"

    # geo_df = image_to_shapefile(
    #     img_path,
    #     shapefile_path=shapefile_path,
    #     visualize=True,
    #     invert=False
    # )
    
    #---------
    #image variables    
    image_basename = "baseland"
    version = "2"
    image_date = "03312025"
    image_name = f"{image_basename}_{image_date}_{version}.jpg" #March 31st
    img_path = directories.IMAGES_DIR / image_name
    
    extract_images.display_image(img_path, title=f'Original Image, {image_name}')
    #---------
    #metadata used in processing- later as systems develop move this to external .config files or another system for storing world processing parameters for all worlds not just Htrea
    feature_types = ['baseland', 'inland_lakes_seas']
    z_levels = [0, 0]
    inversions = [False, True]#false- land (white), true-  water (white)
    flood_fill = [False, True]#false- leave ocean as is, true- turns the ocean (0, 0) black (val-0) (removes ocean and anything connected to it)
    
    #built in processing
    geo_dfs = {}
    shapefile_paths = {}
    combined_fig = None #combined figure of all shapefiles being processed
    #modified processing
    completed_idx = []#[0] #deontes the indeces that we have processed, skip these while developing our systems
    #---------
    for idx, (feature_type, z_level, invert, flood_filled) in enumerate(zip(feature_types, z_levels, inversions, flood_fill)):
        print(f"PROCESSING LAYER:\n{feature_type=}\n{z_level=}\n{flood_filled=}\n{invert=}")
        if(idx in completed_idx):
            continue
    
        metadata = {
            "type": feature_type,
            "level": z_level,
            "source": img_path.name,
        }
        
        shapefile_path = directories.SHAPEFILES_DIR / f"{feature_type}_{DATE}.shp"
        html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_{feature_type}_{DATE}.html"
        shapefile_paths[feature_type] = shapefile_path
        
        #run this feature through the image -> shapefile pipeline        
        print(f"PROCESSING {feature_type}")
        geo_dfs[feature_type] = image_to_shapefile(
            img_path,
            invert=invert,
            output=OUTPUT,
            shapefile_path=shapefile_path,
            visualize=VISUALIZE,
            metadata=metadata,
            flood_fill=flood_fill
        )
        print(f"FINISHED processing {feature_type}")

        # 👀 Get name of next feature if it exists
        if idx + 1 < len(feature_types):
            next_feature = feature_types[idx + 1]
            if USER_DEBUG_MODE:
                input(f'Press any key to continue to the next feature type ({next_feature}): ')
        else:
            print("✅ Finished all feature types!")
        
        #individual feature developed viewing
        fig = world_viewer.plot_polygon_shapes_interactive(shapefile_path)
        viewing_util.save_figure_to_html(fig, html_path, open_on_export=True)
        
        #further develop the figure with all features processed
        combined_fig = world_viewer.plot_polygon_shapes_interactive(shapefile_path, combined_fig)
    
    combined_html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_{DATE}_PIPELINE.html"
    viewing_util.save_figure_to_html(combined_fig, combined_html_path, open_on_export=True)
    
    