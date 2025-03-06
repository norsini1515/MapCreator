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
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from mapcreator import directories
import mapcreator.scripts.extract_images as extract_images



def extract_contours(img_array, min_area=5.0):
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
            coords = [(x, y) for x, y in c[:, 0, :]]
            polygon = Polygon(coords)
            
            # flipped_coords = [(x, height - y) for x, y in c[:, 0, :]]
            # polygon = Polygon(flipped_coords)
            
            # Ensure the polygon is valid before adding it
            if polygon.is_valid and polygon.area >= min_area and len(polygon.exterior.coords) >= 3:
                polygons.append(polygon)
            else:
                print("Invalid polygon skipped")
                # print(height-y)
            # polygons.append(Polygon(flipped_coords))
    
    # Visualize contours
    plt.figure(figsize=(10, 10))
    plt.imshow(contour_img, cmap="gray")
    plt.title("Detected Contours")
    plt.axis("off")
    plt.show()
    
    return polygons

def save_shapefile(polygons, output_path):
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
    
    # Ensure only valid polygons are saved
    valid_polygons = [poly for poly in polygons if poly.is_valid]

    # Save polygons to shapefile using Fiona
    with fiona.open(output_path, mode="w", driver="ESRI Shapefile", schema=schema, crs=CRS.from_epsg(4326)) as shapefile:
        for i, polygon in enumerate(valid_polygons):
            shapefile.write({
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [np.array(polygon.exterior.coords).tolist()],
                },
                "properties": {"id": i},
            })
    
    print(f"Shapefile saved to: {output_path}")
    
def visualize_shapefile(shapefile_path):
    """
    Read and visualize a shapefile using GeoPandas.

    Args:
        shapefile_path: Path to the shapefile.

    Returns:
        None
    """
    print('displaying:', shapefile_path.name)
    
    gdf = gpd.read_file(shapefile_path)
    print("Bounds:", gdf.total_bounds)
    
    # Check if the shapefile has valid bounds
    xmin, ymin, xmax, ymax = gdf.total_bounds
    if np.isinf(xmin) or np.isinf(ymin) or np.isinf(xmax) or np.isinf(ymax):
        raise ValueError("Shapefile contains invalid coordinate bounds.")

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue")

    # Try forcing an aspect ratio fix
    try:
        ax.set_aspect("auto")  # Allow dynamic aspect ratio
    except ValueError as e:
        print(f"Warning: Failed to set aspect ratio automatically: {e}")
        ax.set_aspect("equal", adjustable="datalim")  # Fallback

    plt.title(f"Visualization of {shapefile_path.name}")
    plt.show()


def plot_single_polygon(polygon, index=0):
    """
    Plot a single polygon to check if visualization works.
    
    Args:
        polygon: A Shapely Polygon.
        index: Polygon index for labeling.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    x, y = polygon.exterior.xy
    ax.fill(x, y, edgecolor="black", facecolor="lightblue", linewidth=1)
    
    ax.set_title(f"Polygon {index}")
    plt.show()


def debug_plot_shapefile(shapefile_path):
    """
    Plot polygons individually to debug errors when visualizing the shapefile.
    
    Args:
        shapefile_path: Path to the shapefile.
    """
    problem_polygons = []
    
    gdf = gpd.read_file(shapefile_path)

    if gdf.empty:
        raise ValueError("Shapefile is empty or invalid.")

    print(f"Total Polygons: {len(gdf)}")

    for i, row in gdf.iterrows():
        print(f"Plotting Polygon {i}...")
        try:
            plot_single_polygon(gdf.iloc[[i]], i)
        except Exception as e:
            print(f"Error plotting polygon {i}: {e}")
            problem_polygons.append(i)  # Stop at the first failure
    
    return problem_polygons

def plot_all_polygons_matplotlib(polygons):
    """
    Plot all polygons manually using Matplotlib.
    
    Args:
        polygons: List of Shapely Polygons.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i, polygon in enumerate(polygons):
        x, y = polygon.exterior.xy
        ax.plot(x, y, label=f"Polygon {i}")
    
    # ax.legend()
    plt.title("All Polygons - Matplotlib Debugging")
    plt.show()


            
if __name__ == '__main__':
    image_name = "continent_image_03052025.png"
    # Load and preprocess the image
    landmass_base_map = directories.IMAGES_DIR / image_name
    # landmass_base_map = directories.IMAGES_DIR / "landamass_base.jpg"
    img = extract_images.extract_image_from_file(landmass_base_map)
    img = extract_images.prepare_image(img, contrast_factor=2.0)
    img_array = np.array(img).astype(np.uint8) * 255
    extract_images.display_image(f"{image_name}", img_array, output=False)
    
    img_array = extract_images.invert_image(img_array)
    # extract_images.display_image(f"Inverted {image_name}", img_array, output=False)
    
    # debug_img_array = pd.DataFrame(extract_images.image_array_coords(img_array, value=255), columns=['X', 'Y'])

    # Extract contours
    print("Extracting contours...")
    polygons = extract_contours(img_array, min_area=1.0)
    print(f"Number of polygons detected: {len(polygons)}")
    
    # Save contours to a shapefile
    output_path = directories.SHAPEFILES_DIR / "landmass_contours.shp"
    save_shapefile(polygons, output_path)
    
    print(f"Contours saved to {output_path}")
    visualize_shapefile(output_path)
    
    gdf = gpd.read_file(output_path)
    print("Loaded shapefile:\n", gdf)

    
    problem_polygons = debug_plot_shapefile(output_path)
    plot_all_polygons_matplotlib(polygons)
    
    
    
    plot_single_polygon(polygons[12], index=12)
    problematic_polygon = polygons[12]
    print("Polygon #12 Details:")
    print(f" - Area: {problematic_polygon.area:.2f}")
    print(f" - Bounds: {problematic_polygon.bounds}")
    print(f" - Number of Points: {len(problematic_polygon.exterior.coords)}")
    print(f" - Valid: {problematic_polygon.is_valid}")
    print(f" - Coordinates: {list(problematic_polygon.exterior.coords)}")

