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
    return cli_val if cli_val is not None else cfg.get(key)

# import AFTER helpers so the module loads cleanly
from mapcreator.features.tracing.pipeline import extract_all as _extract_all
from mapcreator.features.tracing.pipeline import write_all as _write_all
from mapcreator.globals.logutil import Logger, info, warn, error, success, process_step, setting_config
from mapcreator.globals import directories, configs

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
    width: Optional[int] = typer.Option(None, "--width", help="Grid width (px)", rich_help_panel="World grid"),
    height: Optional[int] = typer.Option(None, "--height", help="Grid height (px)", rich_help_panel="World grid"),
    xmin: Optional[float] = typer.Option(None, "--xmin", help="Extent min X", rich_help_panel="World grid"),
    ymin: Optional[float] = typer.Option(None, "--ymin", help="Extent min Y", rich_help_panel="World grid"),
    xmax: Optional[float] = typer.Option(None, "--xmax", help="Extent max X", rich_help_panel="World grid"),
    ymax: Optional[float] = typer.Option(None, "--ymax", help="Extent max Y", rich_help_panel="World grid"),
    crs:  Optional[str]  = typer.Option(None, "--crs",  help="Output CRS (e.g., 'EPSG:3857')", rich_help_panel="World grid"),
    # Image Preprocessing
    invert:     Optional[bool]  = typer.Option(None, "--invert",     help="Invert after binarize", rich_help_panel="Image Preprocessing"),
    flood_fill: Optional[bool]  = typer.Option(None, "--flood-fill", help="Flood-fill open regions", rich_help_panel="Image Preprocessing"),
    contrast:   Optional[float] = typer.Option(None, "--contrast",   help="Contrast factor", rich_help_panel="Image Preprocessing"),
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

    meta = {
        "log_file": log_path,
        "verbose": _pick(cfg, "verbose", verbose) or False,
        "image": _pick(cfg, "image", image) or None,
        "out_dir": _pick(cfg, "out_dir", out_dir) or None,
        
        "width": _pick(cfg, "width", width) or 1000,
        "height": _pick(cfg, "height", height) or 1000,
        "extent": dict(
            xmin=_pick(cfg, "xmin", xmin) or 0.0,
            ymin=_pick(cfg, "ymin", ymin) or 0.0,
            xmax=_pick(cfg, "xmax", xmax) or 3500.0,
            ymax=_pick(cfg, "ymax", ymax) or 3500.0,
        ),

        "crs": _pick(cfg, "crs", crs) or "EPSG:3857",
        
        "invert": _pick(cfg, "invert", invert) or False,
        "flood_fill": _pick(cfg, "flood_fill", flood_fill) or False,

        "contrast": _pick(cfg, "contrast", contrast) or 2.0,
        "min_area": _pick(cfg, "min_area", min_area) or 5.0,
        "min_points": _pick(cfg, "min_points", min_points) or 3,
    }

    img = Path(_pick(cfg, "image", image))
    out = Path(_pick(cfg, "out_dir", out_dir))

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

def main():
    app()

if __name__ == "__main__":
    main()
