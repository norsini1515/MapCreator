import typer
app = typer.Typer(no_args_is_help=True)

@app.command()
def init(project: str, width:int=1000, height:int=1000):
    """Scaffold folders + config.yml."""
    ...

@app.command()
def trace(image: str, out: str="coastline.geojson"):
    """Extract/hand-assist coastline + islands from the scan."""
    ...

@app.command()
def make_masks(land: str, water: str, out: str="ocean_mask.geojson"):
    """Build ocean mask from land polygons."""
    ...

@app.command()
def rasters(out_dir: str="data/processed"):
    """Emit terrain_class_map.tif and climate_class_map.tif placeholders."""
    ...

if __name__ == "__main__":
    app()