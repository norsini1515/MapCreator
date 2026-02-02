from pathlib import Path
from mapcreator.globals.logutil import (
    info, process_step, error, setting_config, success
)
import geopandas as gpd

def export_gdf(gdf: gpd.GeoDataFrame, path: Path, verbose:bool|str=False) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    driver = "GeoJSON" if ext == ".geojson" else "ESRI Shapefile" if ext == ".shp" else None
    
    if not driver:
        raise ValueError(f"Unsupported extension: {ext}")
    if verbose in (True, "info", "debug"):
        process_step(f"Exporting GeoDataFrame to {path} using driver {driver}...")
    
    gdf.to_file(path, driver=driver)

def export_gdfs(gdf_dict: dict[str, gpd.GeoDataFrame], out_dir: Path, verbose:bool|str=False) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    for name, gdf in gdf_dict.items():
        path = out_dir / f"{name}.geojson"
        export_gdf(gdf, path, verbose=verbose)
