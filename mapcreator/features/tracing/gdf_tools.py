import pandas as pd
import geopandas as gpd


def to_gdf(polygons, metadata=None, crs="EPSG:3857"):
    metadata = metadata or {}
    df = pd.DataFrame(metadata, index=range(len(polygons)))
    gdf = gpd.GeoDataFrame(df, geometry=polygons, crs=crs)
    gdf["id"] = gdf.index
    return gdf


def dissolve_class(gdf: gpd.GeoDataFrame, class_col="class"):
    return gdf.dissolve(by=class_col, as_index=False)

def _ensure_crs(gdf, crs):
    if gdf.crs is None:
        raise ValueError("merged_gdf.crs is None; set it before rasterizing.")
    if str(gdf.crs) != str(crs):
        # safer to reproject than fail
        return gdf.to_crs(crs)
    return gdf