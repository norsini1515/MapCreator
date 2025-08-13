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