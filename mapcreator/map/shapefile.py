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
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
import plotly.graph_objects as go
from pathlib import Path

from mapcreator import directories, configs
import mapcreator.scripts.extract_images as extract_images
from mapcreator.visualization import polygon_viewer

MIN_AREA = 5
MIN_POINTS = 4
date = datetime.now().strftime('%m%d%y')

def extract_contours(img_array, min_area=MIN_AREA, min_points=MIN_POINTS):
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
    height = img_array.shape[0]  # Get image height for flipping y-coordinates

    for i, c in enumerate(contours):
        if len(c) >= min_points:  # Ensure enough points for a polygon
            coords = [(x, y) for x, y in c[:, 0, :]]

            polygon = Polygon(coords)

            # Apply enhanced filtering
            if polygon.is_valid and polygon.area >= min_area and len(polygon.exterior.coords) >= min_points:
                if polygon.length > min_area * 0.5:  # Avoid extremely thin polygons
                    polygons.append(polygon)
                else:
                    print(f"⚠️ Skipping THIN Polygon {i} | Area: {polygon.area:.2f} | Length: {polygon.length:.2f}")
            else:
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
                print(f'{len(candidates)} candidates for polygon {i}.')
                for j, poly in enumerate(candidates):
                    if poly.is_valid and poly.area >= min_area and len(poly.exterior.coords) >= min_points:
                        polygons.append(poly)
                        print(f"✅ Polygon {i}.{j} repaired using buffer(0) and added.")
                    else:
                        print(f"⚠️ Polygon {i}.{j} from repair was skipped (invalid or too small).")
            print()
            
    return polygons

def save_shapefile(polygons, output_path, type_label="unknown"):
    """
    Save a list of Shapely Polygons to a shapefile with a type classification.

    Args:
        polygons (List[Polygon]): List of Shapely Polygon objects to save.
        output_path (Path): File path to save the resulting shapefile.
        type_label (str): Classification label for the polygons (default: 'unknown').

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
                    "type": type_label,
                },
            })

    print(f"✅ Shapefile saved to: {output_path}")
   
def visualize_shapefile(shapefile_path):
    """
    Enhanced visualization of a shapefile using GeoPandas and Matplotlib.
    """
    print('Displaying:', shapefile_path.name)
    
    gdf = gpd.read_file(shapefile_path)
    print("Bounds:", gdf.total_bounds)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    if np.isinf(xmin) or np.isinf(ymin) or np.isinf(xmax) or np.isinf(ymax):
        raise ValueError("Shapefile contains invalid coordinate bounds.")

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue", aspect=None)

    # Manual aspect ratio
    width = xmax - xmin
    height = ymax - ymin
    if width > 0 and height > 0:
        ax.set_aspect(width / height)
    else:
        ax.set_aspect("equal", adjustable="datalim")

    # Add labels/grid/frame
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Optional polygon labeling (if IDs present)
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, str(row.get("id", idx)), fontsize=8, ha="center", color="black")

    # Optional bounding box outline
    bbox = plt.Rectangle((xmin, ymin), width, height, 
                         linewidth=1, edgecolor='gray', facecolor='none', linestyle=':')
    ax.add_patch(bbox)

    plt.title(f"Visualization of {shapefile_path.stem}")
    
    ax.invert_yaxis()  # Flip Y so that bottom-left is (0,0)
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

def plot_polygon_shapes_interactive(shapefile_path):
    """
    Plots raw 2D polygon shapes from a shapefile using Plotly (no map projection).
    Perfect for fantasy or image-based maps.
    """
    gdf = gpd.read_file(shapefile_path)

    fig = go.Figure()

    for i, row in gdf.iterrows():
        poly = row.geometry
        if poly.is_empty:
            continue
        if poly.geom_type == "Polygon":
            x, y = map(list, poly.exterior.xy)
            
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                fill="toself",
                name=f"ID {row.get('id', i)}",
                hovertext=f"ID: {row.get('id', i)}<br>Type: {row.get('type', 'unknown')}",
                hoverlabel=dict(namelength=0, bgcolor="white", font_size=12),
                hoverinfo="text",
                line=dict(color="#556b2f", width=1),
                fillcolor="rgba(160, 184, 120, 0.5)"
            ))
            
        elif poly.geom_type == "MultiPolygon":
            for j, subpoly in enumerate(poly.geoms):
                x, y = map(list, subpoly.exterior.xy)
            
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines",
                    fill="toself",
                    name=f"ID {row.get('id', i)}.{j}",
                    hovertext=f"ID: {row.get('id', i)}.{j}<br>Type: {row.get('type', 'unknown')}",
                    hoverinfo="text",
                    line=dict(color="#556b2f", width=1),
                    fillcolor="rgba(160, 184, 120, 0.5)"
                ))

    fig.update_layout(
        title="Htrea - Polygon Viewer",
        xaxis_title="X",
        yaxis_title="Y",
        # xaxis=dict(scaleanchor="y", scaleratio=1),
        # yaxis=dict(autorange="reversed"),  # match image space
        yaxis=dict(scaleanchor=None),
        xaxis=dict(scaleanchor=None),
        plot_bgcolor="#f4f1e1",
        paper_bgcolor="#f4f1e1",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    fig.update_yaxes(autorange='reversed')

    return fig




def save_figure_to_html(fig, output_html):
    """
    Saves a Plotly figure to an interactive HTML file.

    Args:
        fig (plotly.graph_objs.Figure): The figure to save.
        output_html (Path or str): Output path.
    """
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))
    print(f"✅ Saved HTML to: {output_html}")
    
if __name__ == '__main__':
    version = "2"
    image_date = "03312025"

    processing_level = 'land'
    processing_group = 'baseland'

    image_name = f"{processing_group}_{image_date}_{version}.jpg" #March 31st
    
    # Load and preprocess the image
    landmass_base_map = directories.IMAGES_DIR / image_name
    # landmass_base_map = directories.IMAGES_DIR / "landamass_base.jpg"
    img = extract_images.extract_image_from_file(landmass_base_map)
    img_array = extract_images.invert_image(extract_images.prepare_image(img, contrast_factor=2.0))
    
    # img_array_filled = fill_water_regions_to_array(landmass_base_map)
    # extract_images.display_image(f"Inverted {image_name}", img_array_filled, output=False)
    
    extract_images.display_image(f"{image_name[:-4]}", img_array, output=True, resize_dim=img_array.shape)
    
    # Extract contours
    print("Extracting contours...")
    polygons = extract_contours(img_array, min_area=1.0)
    # plot_all_polygons(polygons, title='The world of Htrea')
    print(f"Number of polygons detected: {len(polygons)}")
    
    # Save contours to a shapefile
    output_path = directories.SHAPEFILES_DIR / f"{processing_level}_{date}.shp"
    save_shapefile(polygons, output_path, processing_level)
    
    visualize_shapefile(output_path)
    
    interactive_world_file = f"{configs.WORLD_NAME}_{processing_level}_{date}.html"
    # fig = build_interactive_map(output_path)
    fig = plot_polygon_shapes_interactive(output_path)
    save_figure_to_html(fig, directories.DATA_DIR / interactive_world_file)
    
    ###########################################################################
    #Was used in debugging stage of extract_countours
    if False:
        gdf = gpd.read_file(output_path)
        print("Loaded shapefile:"); gdf.info()
        
        problematic_polygons = polygon_viewer.debug_shapefile_interactive(output_path)
        
        if(len(problematic_polygons)):
            #save all problematic polygons 
            save_shapefile(polygons, output_path, problematic_polygons)
            #print information on any problematic polygons.
            for i in problematic_polygons:
                polygon = polygons[i]  # Retrieve actual Polygon using index
                polygon_viewer.plot_single_polygon(polygon, index=i)
                
                print(f"Polygon #{i} Details:")
                print(f" - Area: {polygon.area:.2f}")
                print(f" - Bounds: {polygon.bounds}")
                print(f" - Number of Points: {len(polygon.exterior.coords)}")
                print(f" - Valid: {polygon.is_valid}")
                print(f" - Coordinates: {list(polygon.exterior.coords)}")