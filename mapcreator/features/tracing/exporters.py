from pathlib import Path

def export_gdf(gdf, path: Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    driver = "GeoJSON" if ext == ".geojson" else "ESRI Shapefile" if ext == ".shp" else None
    if not driver:
        raise ValueError(f"Unsupported extension: {ext}")
    gdf.to_file(path, driver=driver)
    return path
