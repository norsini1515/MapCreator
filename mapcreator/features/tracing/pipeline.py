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
from .rasters import init_empty_raster_pair
from .water_classify import split_ocean_vs_inland, inland_mask_to_polygons


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


def extract_all(image: Path, out_dir: Path, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    land_gdf, bin_img = image_to_land_gdf(image, meta)
    ocean_gdf, waterb_gdf = water_gdfs_from_binary(bin_img, meta)

