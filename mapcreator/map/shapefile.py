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
import geopandas as gpd
from fiona.crs import CRS
from datetime import datetime
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
from pathlib import Path

from mapcreator import directories, configs
import mapcreator.scripts.extract_images as extract_images
from mapcreator import polygon_viewer, world_viewer
from mapcreator.visualization import viewing_util

def extract_contours(img_array, min_area=configs.MIN_AREA, min_points=configs.MIN_POINTS,
                     verbose=True):
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

def save_shapefile(polygons, output_path, classification_label="unknown"):
    """
    Save a list of Shapely Polygons to a shapefile with a type classification.

    Args:
        polygons (List[Polygon]): List of Shapely Polygon objects to save.
        output_path (Path): File path to save the resulting shapefile.
        classification_label (str): Classification label for the polygons (default: 'unknown').

    Returns:
        None
    """
    
    schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "int",
            "type": "str",
        },
    }

    os.makedirs(directories.SHAPEFILES_DIR, exist_ok=True)

    with fiona.open(output_path, mode="w", driver="ESRI Shapefile", schema=schema, crs=CRS.from_epsg(4326)) as shapefile:
        for i, polygon in enumerate(polygons):
            shapefile.write({
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [np.array(polygon.exterior.coords).tolist()],
                },
                "properties": {
                    "id": i,
                    "type": classification_label,
                },
            })

    print(f"✅ Shapefile saved to: {output_path}")
   
def visualize_shapefile(data, title="Shapefile Visualization"):
    """
    Visualizes either a shapefile from path or a list of Shapely polygons.

    Args:
        data (Path or List[Polygon]): Path to a shapefile OR a list of Shapely polygons.
        title (str): Title for the plot.
    """
    if isinstance(data, (str, Path)):
        gdf = gpd.read_file(data)
        source = Path(data).stem
    elif isinstance(data, list):
        gdf = gpd.GeoDataFrame({"geometry": data})
        source = title or "Polygon List"
    else:
        raise TypeError("Input must be a Path or list of Shapely Polygon objects.")

    if gdf.empty:
        raise ValueError("GeoDataFrame is empty, cannot visualize.")

    print(f"🗺️ Displaying: {source}")
    print("Bounds:", gdf.total_bounds)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    if np.isinf(xmin) or np.isinf(ymin) or np.isinf(xmax) or np.isinf(ymax):
        raise ValueError("Shapefile contains invalid coordinate bounds.")

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue", aspect=None)

    width = xmax - xmin
    height = ymax - ymin
    ax.set_aspect(width / height if width > 0 and height > 0 else "equal", adjustable="datalim")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # for idx, row in gdf.iterrows():
    #     if row.geometry.is_empty:
    #         continue
    #     centroid = row.geometry.centroid
    #     ax.text(centroid.x, centroid.y, str(row.get("id", idx)), fontsize=8, ha="center", color="black")

    bbox = plt.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='gray', facecolor='none', linestyle=':')
    ax.add_patch(bbox)

    ax.invert_yaxis()
    plt.title(f"Visualization of {source}")
    plt.show()
        
def fill_water_regions_to_array(img_path: Path) -> np.ndarray:
    """
    Converts a landmask image (white land, black water) into a filled landmass array.
    Useful for contour extraction of the full base landmass.

    Args:
        img_path: Path to the original 'LakesOPEN' binary image.

    Returns:
        Numpy array of the filled landmass image (binary: 0 for background, 255 for land)
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    # Ensure the image is binary (in case of compression artifacts)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Invert: water becomes white (255), land black (0)
    inverted = cv2.bitwise_not(binary)

    # Flood fill background from top-left corner (assumed sea)
    h, w = inverted.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inverted, mask, (0, 0), 0)  # Fill sea with black (0)

    # Invert back: land is white, lakes/seas filled in
    filled = cv2.bitwise_not(inverted)

    return filled

def image_to_shapefile(
    img_path: Path,
    invert: bool = False,
    output: bool = False,
    output_path: Path = None,
    visualize: bool=True, 
    contrast_factor: float=2.0, 
    min_area: float=1.0, 
    min_points: int=3, 
    processing_level: int=0,
    processing_group: str="Unknown",
    ):
    """
    End-to-end pipeline to convert an image to a shapefile of extracted polygons.

    Args:
        img_path (Path): Path to the input .jpg image.
        invert (bool): Whether to invert image after binarization.
        output (bool): If True, save the shapefile.
        output_path (Path): Where to save the shapefile if output=True.
        visualize (bool): Whether to visualize the shapefile.
        contrast_factor (float): Contrast enhancement factor.
        min_area (float): Minimum polygon area to keep.
        min_points (int): Minimum number of points per polygon.
        # processing_level (str): Tag for saving shapefile metadata.

    Returns:
        Path or List[Polygon]: Output path or list of polygons.
    """
    if(not str(img_path).endswith('.jpg')):
        raise ValueError('expecting a .jpg input image')
        
    img = extract_images.extract_image_from_file(img_path)
    img_array = extract_images.prepare_image(img, contrast_factor=contrast_factor)
        
    
    if invert:
        img_array = extract_images.invert_image(img_array)
        
    # Extract contours
    print("Extracting contours...")
    polygons = extract_contours(img_array, min_area, min_points, verbose=False)
    print(f"✅ Detected {len(polygons)} polygons.")
    
    if output and output_path is None:
       output_filename = f"{processing_group}_{date}.shp"
       output_path = directories.SHAPEFILES_DIR / output_filename
    
    if visualize:
        visualize_shapefile(polygons)
        
    if output:
        save_shapefile(polygons, output_path, processing_group)
    
    return {
    "polygons": polygons,
    "shapefile_path": output_path if output else None,
    }

if __name__ == '__main__':
    date = datetime.now().strftime('%m%d%y')
    
    version = "2"
    image_date = "03312025"
    processing_level = 0
    processing_group = 'baseland'

    image_name = f"{processing_group}_{image_date}_{version}.jpg" #March 31st
    img_path = directories.IMAGES_DIR / image_name
    
    shapefile_path = directories.SHAPEFILES_DIR / f"{processing_group}_{date}.shp"
    html_path = directories.DATA_DIR / f"{configs.WORLD_NAME}_{processing_level}_{date}.html"
    
    #Full pipeline
    polygons = image_to_shapefile(
        img_path,
        invert=False,
        output=True,
        output_path=shapefile_path,
        visualize=True,
        contrast_factor=2.0,
        min_area=1.0,
        min_points=3,
        processing_level=processing_level,
        processing_group=processing_group,
    )['polygons']
    
    fig = world_viewer.plot_polygon_shapes_interactive(shapefile_path)
    viewing_util.save_figure_to_html(fig, html_path)