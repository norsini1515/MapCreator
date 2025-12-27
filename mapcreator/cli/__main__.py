# mapcreator/cli/__main__.py
from pathlib import Path
from typing import Optional, Dict, Any
import typer, yaml
from datetime import datetime

# --- Typer app (root has no options) ---
app = typer.Typer(no_args_is_help=True)

def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        info(f"Using config: {path}")
        return cfg
    except FileNotFoundError:
        error(f"(config not found at {path}; using flags/defaults)")
        return {}

def _pick(cfg: Dict[str, Any], key: str, cli_val):
    config_val = cfg.get(key)
    if key=='verbose':
        print(f"pick {key}: cli_val={cli_val}, cfg.get={config_val}")
    return config_val if config_val is not None else cli_val

# import AFTER helpers so the module loads cleanly
from mapcreator.features.tracing.pipeline import write_all as _write_all
from mapcreator.globals.logutil import Logger, info, warn, error, success, process_step, setting_config
from mapcreator.globals import directories, configs
from mapcreator.features.tracing.reclass import burn_polygons_into_class_raster, apply_palette_from_yaml

# DEFAULT_CONFIG_FILE_PATH
DEFAULT_CONFIG_FILE_PATH = directories.CONFIG_DIR / configs.IMAGE_TRACING_EXTRACT_CONFIGS_NAME
# LOG_FILE_NAME
LOG_FILE_NAME =  f"{configs.WORLD_NAME.replace(' ', '_')}_extract.log"


def _resolve_log_path() -> Path:
    '''
    Resolve the log file path from the config or defaults.
    '''
    # default under out_dir/logs (fallback to project ./logs if out_dir missing)
    base = directories.LOGS_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{LOG_FILE_NAME}{datetime.now():%Y%m%d_%H%M%S}.log"

# --- Helper: unified config loading ---
def load_config(cfg: Dict[str, Any], **cli_values) -> Dict[str, Any]:
    """Build the unified configuration metadata dict.

    This mirrors the logic previously embedded in ``extract_all`` so other
    commands can reuse it. Precedence currently matches existing behavior:
    if a key exists in the YAML it overrides the CLI value; otherwise the
    CLI value (or fallback default) is used.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Parsed YAML configuration (may be empty).
    **cli_values : Any
        Keyword arguments representing CLI-provided values. Expected keys:
        verbose, image, out_dir, width, height, xmin, ymin, xmax, ymax, crs,
        invert, flood_fill, contrast, min_area, min_points, log_file.

    Returns
    -------
    Dict[str, Any]
        Normalized configuration dictionary used by processing pipelines.
    """
    def pick(key: str, cli_val):
        config_val = cfg.get(key)
        return config_val if config_val is not None else cli_val

    meta = {
        "log_file": pick("log_file", cli_values.get("log_file")),
        "verbose": pick("verbose", cli_values.get("verbose")) or False,
        "image": pick("image", cli_values.get("image")) or None,
        "out_dir": pick("out_dir", cli_values.get("out_dir")) or None,
        # "width": pick("width", cli_values.get("width")) or 1000,
        # "height": pick("height", cli_values.get("height")) or 1000,
        "extent": dict(
            xmin=pick("xmin", cli_values.get("xmin")) or 0.0,
            ymin=pick("ymin", cli_values.get("ymin")) or 0.0,
            xmax=pick("xmax", cli_values.get("xmax")) or 3500.0,
            ymax=pick("ymax", cli_values.get("ymax")) or 3500.0,
        ),
        "crs": pick("crs", cli_values.get("crs")) or "EPSG:3857",
        # "invert": pick("invert", cli_values.get("invert")) or False,
        # "flood_fill": pick("flood_fill", cli_values.get("flood_fill")) or False,
        # "contrast": pick("contrast", cli_values.get("contrast")) or 2.0,
        "min_area": pick("min_area", cli_values.get("min_area")) or 5.0,
        "min_points": pick("min_points", cli_values.get("min_points")) or 3,
    }
    return meta

# --- SUBCOMMAND ---
@app.command("extract-all")
def extract_all(
    # I/O
    config: Optional[Path] = typer.Option(
        DEFAULT_CONFIG_FILE_PATH, "--config", "-c",
        help="YAML file with parameters. Flags override YAML.",
        rich_help_panel="I/O"
    ),
    image: Optional[Path] = typer.Option(
        None, "--image", "-i",
        help="Input raster (jpg/png). If omitted, read from YAML.",
        rich_help_panel="I/O"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", "-o",
        help="Output directory. If omitted, read from YAML.",
        rich_help_panel="I/O"
    ),
    # World grid / extent
    xmin: Optional[float] = typer.Option(None, "--xmin", help="Extent min X", rich_help_panel="World grid"),
    ymin: Optional[float] = typer.Option(None, "--ymin", help="Extent min Y", rich_help_panel="World grid"),
    xmax: Optional[float] = typer.Option(None, "--xmax", help="Extent max X", rich_help_panel="World grid"),
    ymax: Optional[float] = typer.Option(None, "--ymax", help="Extent max Y", rich_help_panel="World grid"),
    crs:  Optional[str]  = typer.Option(None, "--crs",  help="Output CRS (e.g., 'EPSG:3857')", rich_help_panel="World grid"),
    # Image Preprocessing
    # invert:     Optional[bool]  = typer.Option(None, "--invert",     help="Invert after binarize", rich_help_panel="Image Preprocessing"),
    # flood_fill: Optional[bool]  = typer.Option(None, "--flood-fill", help="Flood-fill open regions", rich_help_panel="Image Preprocessing"),
    # contrast:   Optional[float] = typer.Option(None, "--contrast",   help="Contrast factor", rich_help_panel="Image Preprocessing"),
    # Geometry filters
    min_area:   Optional[float] = typer.Option(None, "--min-area",   help="Min polygon area (pre-affine)", rich_help_panel="Geometry filters"),
    min_points: Optional[int]   = typer.Option(None, "--min-points", help="Min ring vertices", rich_help_panel="Geometry filters"),
    # Utility
    dry_run: bool = typer.Option(False, "--dry-run", help="Print resolved config and exit", rich_help_panel="Utility"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging", rich_help_panel="Utility"),
):
    '''
    Extract all features from a raster image.
    The main processing steps include:
    1. Loading the image
    2. Applying preprocessing
    3. Extracting features
    4. Saving the results

    Optionals:

    '''
    # Set up the logging configuration
    log_path = _resolve_log_path()
    logger = Logger(logfile_path=log_path)
    # load configurations
    cfg = _load_yaml(config)
    meta = load_config(
        cfg,
        log_file=log_path,
        verbose=verbose,
        image=image,
        out_dir=out_dir,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        crs=crs,
        # invert=invert,
        # flood_fill=flood_fill,
        # contrast=contrast,
        min_area=min_area,
        min_points=min_points,
    )
    info("Resolved configuration:")
    for k, v in meta.items():
        setting_config(f"  {k}: {v}")

    # Resolve image and out dir from meta (already merged)
    img = Path(meta["image"]) if meta["image"] else None
    out = Path(meta["out_dir"]) if meta["out_dir"] else None

    if dry_run:
        info("Resolved configuration:")
        for k, v in meta.items(): setting_config(f"  {k}: {v}")
        raise typer.Exit()

    if not img or not out:
        error("image and out_dir are required (via YAML or flags).")
        raise typer.Exit(code=2)

    out.mkdir(parents=True, exist_ok=True)
    results = _write_all(img, out, meta)
    for k, v in results.items():
        info(f"{k}: {v}")
    success(f"Extraction completed successfully!, Outputs in: {out}")

    logger.teardown()

@app.command("test")
def test():
    success("Test command executed successfully!")

@app.command("image-extract")
def image_extract(
    # I/O
    config: Optional[Path] = typer.Option(
        DEFAULT_CONFIG_FILE_PATH, "--config", "-c",
        help="YAML file with parameters. Flags override YAML.",
        rich_help_panel="I/O"
    ),
):
    """turn a hand drawn map into a black and white outline image ready for further processing or easy manipulation"""
    
    
    
    success("Image extract command executed successfully!")

def main():
    app()

@app.command("recolor")
def recolor(
    raster: Path = typer.Argument(..., help="Path to class raster (world/terrain/climate)."),
    section: str = typer.Option("terrain", "--section", "-s", help="Section in YAML: base|terrain|climate"),
    config: Path = typer.Option(directories.CONFIG_DIR / "raster_classifications.yml", "--config", "-c", help="YAML file for class colors."),
):
    """Reapply palette from YAML to an existing class raster (in place)."""
    cfg = _load_yaml(config)
    apply_palette_from_yaml(raster, cfg, section)

@app.command("paint")
def paint(
    raster: Path = typer.Argument(..., help="Path to terrain/climate class raster to edit."),
    polygons: Path = typer.Argument(..., help="Vector file with polygons to burn (GeoJSON/Shapefile)."),
    label: str = typer.Option(..., "--label", "-l", help="Class label or ID to assign (e.g., 'mountain' or 10)."),
    section: str = typer.Option("terrain", "--section", "-s", help="Section in YAML: base|terrain|climate"),
    config: Path = typer.Option(directories.CONFIG_DIR / "raster_classifications.yml", "--config", "-c", help="YAML file for class ids/colors."),
    output: Path = typer.Option(None, "--output", "-o", help="Optional output path; if omitted, writes <name>_edited.tif"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Edit raster in place."),
):
    """Burn polygons into an existing class raster as a specific class label/ID."""
    cfg = _load_yaml(config)
    burn_polygons_into_class_raster(
        raster_path=raster,
        polygons_path=polygons,
        class_config=cfg,
        section=section,
        label_or_id=label,
        output=output,
        overwrite=overwrite,
    )

if __name__ == "__main__":
    main()