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
from fiona.crs import from_epsg
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from mapcreator import directories
import mapcreator.scripts.extract_images as extract_images



def extract_contours(img_array):
    """
    Extract contours from a binary image.

    Args:
        img_array: NumPy array representing a binary image.

    Returns:
        List of Shapely Polygons representing contours.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a blank image
    contour_img = np.zeros_like(img_array)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
    
    extract_images.display_image("Contour Drawing Image", contour_img, output=False)
    
    if not contours:
        raise ValueError("No contours were found in the provided image.")
        
    # Convert contours to Shapely Polygons
    # polygons = [Polygon(c[:, 0, :]) for c in contours if len(c) > 2]
    
    # Convert contours to Shapely Polygons and adjust orientation by flipping y-coordinates
    polygons = []
    height = img_array.shape[0]  # Get the height of the image (number of rows)
    for c in contours:
        if len(c) > 2:  # Ensure there are enough points for a polygon
            # Flip y-coordinates (to match GIS system)
            flipped_coords = [(x, height - y) for x, y in c[:, 0, :]]
            polygon = Polygon(flipped_coords)
            # Ensure the polygon is valid before adding it
            if polygon.is_valid:
                polygons.append(polygon)
            else:
                print(f"Invalid polygon skipped: {flipped_coords}")
                # print(height-y)
            # polygons.append(Polygon(flipped_coords))
    
    # Visualize contours
    plt.figure(figsize=(10, 10))
    plt.imshow(contour_img, cmap="gray")
    plt.title("Detected Contours")
    plt.axis("off")
    plt.show()
    
    return polygons

def visualize_shapefile(shapefile_path):
    """
    Read and visualize a shapefile using GeoPandas.

    Args:
        shapefile_path: Path to the shapefile.

    Returns:
        None
    """
    gdf = gpd.read_file(shapefile_path)
    gdf.plot(edgecolor="black", facecolor="lightblue", figsize=(10, 10))
    plt.title(f"Visualization of {shapefile_path.name}")
    plt.show()

def save_shapefile(polygons, output_path, visualize=True):
    """
    Save a list of Shapely Polygons to a shapefile.

    Args:
        polygons: List of Shapely Polygons to save.
        output_path: Path to the output shapefile.
    """
    
    os.makedirs(directories.SHAPEFILES_DIR, exist_ok=True)
    
    # Define schema for shapefile
    schema = {
        "geometry": "Polygon",
        "properties": {"id": "int"},
    }

    # Save polygons to shapefile using Fiona
    with fiona.open(output_path, mode="w", driver="ESRI Shapefile", schema=schema, crs=from_epsg(4326)) as shapefile:
        for i, polygon in enumerate(polygons):
            shapefile.write({
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [np.array(polygon.exterior.coords).tolist()],
                },
                "properties": {"id": i},
            })
    
    print(f"Shapefile saved to: {output_path}")
    
    if visualize:
        visualize_shapefile(output_path)
            
if __name__ == '__main__':
    # Load and preprocess the image
    landmass_base_map = directories.IMAGES_DIR / "landamass_base.jpg"
    img = extract_images.extract_image_from_file(landmass_base_map)
    img = extract_images.prepare_image(img, contrast_factor=2.0)
    img_array = np.array(img).astype(np.uint8) * 255
    
    extract_images.display_image("Preprocessed Image", img_array, output=False)
    
    # debug_img_array = pd.DataFrame(extract_images.image_array_coords(img_array, value=255), columns=['X', 'Y'])

    # Extract contours
    print("Extracting contours...")
    polygons = extract_contours(img_array)
    print(f"Number of polygons detected: {len(polygons)}")
    
    # Save contours to a shapefile
    output_path = directories.SHAPEFILES_DIR / "landmass_contours.shp"
    save_shapefile(polygons, output_path)
    
    print(f"Contours saved to {output_path}")