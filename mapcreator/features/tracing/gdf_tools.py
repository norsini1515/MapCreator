import pandas as pd, geopandas as gpd

def to_gdf(polygons, metadata=None, crs="EPSG:4326"):
    metadata = metadata or {}
    df = pd.DataFrame(metadata, index=range(len(polygons)))
    gdf = gpd.GeoDataFrame(df, geometry=polygons, crs=crs)
    gdf["id"] = gdf.index
    return gdf