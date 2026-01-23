import pandas as pd
import geopandas as gpd
from typing import List

from mapcreator.globals.logutil import (
    info, process_step, error, setting_config, success, warn
)

def to_gdf(polygons, metadata=None, crs="EPSG:3857"):
    metadata = metadata or {}
    df = pd.DataFrame(metadata, index=range(len(polygons)))
    gdf = gpd.GeoDataFrame(df, geometry=polygons, crs=crs)
    gdf["id"] = gdf.index
    return gdf

def dissolve_class(gdf: gpd.GeoDataFrame, class_col:str="class"):
    '''class_col: column name to dissolve by'''
    return gdf.dissolve(by=class_col, as_index=False)

def _ensure_crs(gdf, crs):
    if gdf.crs is None:
        raise ValueError("merged_gdf.crs is None; set it before rasterizing.")
    if str(gdf.crs) != str(crs):
        # safer to reproject than fail
        return gdf.to_crs(crs)
    return gdf

def merge_gdfs(gdfs: gpd.GeoDataFrame|List[gpd.GeoDataFrame], 
               verbose:bool = False,
            ) -> gpd.GeoDataFrame:
    
    """Merge multiple GeoDataFrames into one, ensuring non-empty and consistent CRS."""

    if isinstance(gdfs, gpd.GeoDataFrame):
        gdfs = [gdfs]
    elif not isinstance(gdfs, list):
        raise ValueError("gdfs must be a GeoDataFrame or list of GeoDataFrames.")
    
    valid_gdfs = [gdf for gdf in gdfs if not gdf.empty]
    crs=valid_gdfs[0].crs
    if verbose:
        info(f"Merging {len(valid_gdfs)} GeoDataFrames into one with CRS {crs}.")
        if len(valid_gdfs) < len(gdfs):
              warn(f"{len(valid_gdfs)}/{len(gdfs)} frames valid...")

    merged_gdf = gpd.GeoDataFrame(
        pd.concat(valid_gdfs, ignore_index=True),
        crs=crs,
    )

    if verbose:
        from matplotlib import pyplot as plt

        info(f"Merged GDF: {len(merged_gdf)} features, CRS: {merged_gdf.crs}, shape {merged_gdf.shape}\n")
        merged_gdf.plot(column="class", legend=True)
        plt.show()

    return merged_gdf

