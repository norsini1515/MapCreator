# mapcreator/tracing/pipeline.py
from pathlib import Path
from .img_preprocess import preprocess_image
from .contour_extraction import find_external_contours
from .polygonize import contours_to_polygons
from .gdf_tools import to_gdf
from .geo_transform import pixel_affine, apply_affine_to_gdf

def _image_to_gdf(img: Path, meta: dict, extent: tuple):
    bin_img   = preprocess_image(img,
                                 contrast_factor=meta.get("contrast", 2.0),
                                 invert=meta.get("invert", False),
                                 flood_fill=meta.get("flood_fill", False))
    contours  = find_external_contours(bin_img, tree=meta.get("use_tree", False))
    polys     = contours_to_polygons(contours,
                                     min_area=meta.get("min_area", 5.0),
                                     min_points=meta.get("min_points", 3),
                                     allow_holes=meta.get("allow_holes", True))
    gdf       = to_gdf(polys, metadata=meta, crs="EPSG:3857")
    A         = pixel_affine(width=bin_img.shape[1], height=bin_img.shape[0], **meta.get("extent", {"xmin":0,"ymin":0,"xmax":1000,"ymax":1000}))
    return apply_affine_to_gdf(gdf, A)

def image_to_land(img: Path, meta: dict) -> "GeoDataFrame":
    return _image_to_gdf(img, meta, extent=meta.get("extent"))

def image_to_lakes(img: Path, meta: dict) -> "GeoDataFrame":
    return _image_to_gdf(img, meta, extent=meta.get("extent"))