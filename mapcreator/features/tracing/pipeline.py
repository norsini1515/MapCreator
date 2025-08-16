from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from .img_preprocess import preprocess_image
from .contour_extraction import find_external_contours
# from .polygonize import contours_to_polygons  # keep simple for land shells
from .polygonize import contours_to_polygons
from .gdf_tools import to_gdf, dissolve_class
from .geo_transform import pixel_affine, apply_affine_to_gdf
from .exporters import export_gdf
from .rasters import make_land_water_masks
from .water_classify import split_ocean_vs_inland, inland_mask_to_polygons

from mapcreator.globals.logutil import info, process_step, error


def _affine(meta):
    return pixel_affine(meta["width"], meta["height"], **meta["extent"])


def image_to_land_gdf(img_path: Path, meta: dict):
    bin_img = preprocess_image(
        img_path,
        contrast_factor=meta.get("contrast", 2.0),
        invert=meta.get("invert", False),
        flood_fill=meta.get("flood_fill", False),
    )
    contours = find_external_contours((bin_img > 0).astype("uint8") * 255)
    polys = contours_to_polygons(contours, min_area=meta.get("min_area", 5.0), min_points=meta.get("min_points", 3))
    land = to_gdf(polys, {"class": "land"}, crs=meta["crs"]) if polys else gpd.GeoDataFrame(columns=["class", "geometry"], crs=meta["crs"]) 
    A = _affine(meta)
    return apply_affine_to_gdf(land, A), bin_img


def water_gdfs_from_binary(bin_img: np.ndarray, meta: dict):
    water_mask = (bin_img == 0).astype("uint8")
    ocean_mask, inland_mask = split_ocean_vs_inland(water_mask)

    # Ocean as simple shells (external contours)
    from .contour_extraction import contours_from_binary_mask
    oc_contours = contours_from_binary_mask(ocean_mask)
    oc_polys = contours_to_polygons(oc_contours, min_area=meta.get("min_area", 5.0), min_points=meta.get("min_points", 3))

    # Inland with holes (nested islands)
    in_polys = inland_mask_to_polygons(inland_mask, min_area=meta.get("min_area", 5.0), min_points=meta.get("min_points", 3))

    ocean = to_gdf(oc_polys, {"class": "ocean"}, crs=meta["crs"]) if oc_polys else gpd.GeoDataFrame(columns=["class", "geometry"], crs=meta["crs"]) 
    waterb = to_gdf(in_polys, {"class": "waterbody"}, crs=meta["crs"]) if in_polys else gpd.GeoDataFrame(columns=["class", "geometry"], crs=meta["crs"]) 

    A = _affine(meta)
    return apply_affine_to_gdf(ocean, A), apply_affine_to_gdf(waterb, A)


def build_merged_base(land_gdf: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame, ocean_gdf: gpd.GeoDataFrame):
    parts = [df for df in [land_gdf, water_gdf, ocean_gdf] if not df.empty]
    if not parts:
        return gpd.GeoDataFrame(columns=["class", "geometry"], crs=land_gdf.crs)
    all_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=land_gdf.crs)
    return dissolve_class(all_gdf, class_col="class")


def extract_all(image: Path, meta: dict):
    """
    Extract all relevant features from the input image.
    """
    process_step(f"Extracting features from {image.name}...")
    land_gdf, bin_img = image_to_land_gdf(image, meta)
    ocean_gdf, waterb_gdf = water_gdfs_from_binary(bin_img, meta)
    print(f"Land GDF: {len(land_gdf)} features, Ocean GDF: {len(ocean_gdf)} features, Waterbody GDF: {len(waterb_gdf)} features")
    print(f"Land GDF CRS: {land_gdf.crs}, Ocean GDF CRS: {ocean_gdf.crs}, Waterbody GDF CRS: {waterb_gdf.crs}")
    print(f"Land GDF shape {land_gdf.shape=}, Ocean GDF shape {ocean_gdf.shape}, Waterbody GDP shape {waterb_gdf.shape}\n")

    process_step("Building merged base...")
    merged_gdf = build_merged_base(land_gdf, waterb_gdf, ocean_gdf)
    
    return land_gdf, waterb_gdf, ocean_gdf, merged_gdf, bin_img

def write_all(image: Path, out_dir: Path, meta: dict, *, make_rasters: bool = True):
    """
    Convenience wrapper: run extract_all() and WRITE everything to disk.
    Returns a dict of output paths.
    """
    land_gdf, waterb_gdf, ocean_gdf, merged_gdf, _ = extract_all(image, meta)

    out_dir.mkdir(parents=True, exist_ok=True)

    land_path   = export_gdf(land_gdf,   out_dir / "land.geojson")
    water_path  = export_gdf(waterb_gdf, out_dir / "waterbodies.geojson")
    ocean_path  = export_gdf(ocean_gdf,  out_dir / "ocean.geojson")
    merged_path = export_gdf(merged_gdf, out_dir / "merged_base_geography.geojson")

    land_mask_path, water_mask_path, \
    terrain_class_path, climate_class_path = make_land_water_masks(
        merged_gdf, out_dir,
        width=meta["width"], height=meta["height"],
        extent=meta["extent"], crs=meta["crs"]
    )
    # terrain_path = out_dir / land_mask
    # climate_path = out_dir / "climate_class_map.tif"
    # if make_rasters:
        # init_paintable_class_raster_from_land_mask(land_mask, terrain_path, dtype="uint8", nodata=0)
        # init_paintable_class_raster_from_land_mask(land_mask, climate_path, dtype="uint8", nodata=0)

    return {
        "land": str(land_path),
        "waterbodies": str(water_path),
        "ocean": str(ocean_path),
        "merged": str(merged_path),
        "terrain": str(terrain_class_path),
        "climate": str(climate_class_path),
    }