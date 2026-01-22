from pathlib import Path
from mapcreator.globals.logutil import info, process_step, error, setting_config, success

def export_gdf(gdf, path: Path, verbose:bool|str=False) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    driver = "GeoJSON" if ext == ".geojson" else "ESRI Shapefile" if ext == ".shp" else None
    
    if not driver:
        raise ValueError(f"Unsupported extension: {ext}")
    if verbose in (True, "info", "debug"):
        process_step(f"Exporting GeoDataFrame to {path} using driver {driver}...")
    
    gdf.to_file(path, driver=driver)
