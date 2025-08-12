import numpy as np
import geopandas as gpd
from mapcreator import directories

from shapely.geometry import Point

input_date = "051025"
file_path = directories.DATA_DIR / f"ocean_ray_dataset_{input_date}.geojson"


rays_gdf = gpd.read_file(file_path)
print(rays_gdf.dtypes, rays_gdf.shape, sep='\n')


print()
print(rays_gdf.head(5))

print(rays_gdf['hit_type'].value_counts())

rays_gdf["start_point"] = rays_gdf.geometry.apply(lambda g: Point(g.coords[0]))
rays_gdf["end_point"] = rays_gdf.geometry.apply(lambda g: Point(g.coords[-1]))