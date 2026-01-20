"""
CLI entry point for MapCreator feature extraction pipelines.

commands:
- extract-all: runs the full pipeline (image preprocessing, vector extraction, raster generation)
- extract-image: runs only the image preprocessing step, outputs preprocessed images
- extract-vector: runs only the vector extraction step, outputs GeoJSONs
- extract-raster: runs only the raster generation step, outputs class rasters
To rector:
- recolor: utility to reapply palette from YAML to an existing class raster
- paint: utility to burn polygons into an existing class raster as a specific class label/ID

Usage examples:
- Full pipeline with config file:
    mapc extract-all --config path/to/config.yml

"""
# mapcreator/cli/__main__.py
from pathlib import Path
from typing import Optional, Dict, Any
import typer, yaml
from datetime import datetime

# --- Typer app (root has no options) ---
app = typer.Typer(no_args_is_help=True)

def _pick(cfg: Dict[str, Any], key: str, cli_val):
    config_val = cfg.get(key)
    if key=='verbose':
        print(f"pick {key}: cli_val={cli_val}, cfg.get={config_val}")
    return config_val if config_val is not None else cli_val

# import AFTER helpers so the module loads cleanly
from mapcreator.features.tracing import pipeline as tracing_pipeline
from mapcreator.globals.logutil import Logger, info, warn, error, success, process_step, setting_config
from mapcreator.globals import directories, configs
from mapcreator.globals.config_models import ExtractConfig, read_config_file
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
def load_config(cfg: ExtractConfig, **cli_kwargs) -> Dict[str, Any]:
    """Build the unified configuration metadata dict.

    This mirrors the logic previously embedded in ``extract_all`` so other
    commands can reuse it. Precedence currently matches existing behavior:
    if a key exists in the YAML it overrides the CLI value; otherwise the
    CLI value (or fallback default) is used.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Parsed YAML configuration (may be empty).
    **cli_kwargs : Any
        Keyword arguments representing CLI-provided values. Expected keys:
        verbose, image, out_dir, width, height, xmin, ymin, xmax, ymax, crs,
        invert, flood_fill, contrast, min_area, min_points, log_file.

    Returns
    -------
    Dict[str, Any]
        Normalized configuration dictionary used by processing pipelines.
    """
    def pick(key: str, cli_val):
        config_val = getattr(cfg, key, None)
        return config_val if config_val is not None else cli_val

    meta = {
        #Logging related configs
        "log_file": pick("log_file", cli_kwargs.get("log_file")),
        "verbose": pick("verbose", cli_kwargs.get("verbose")) or False,
        
        #Image/Pipeline processing configs
        "image": pick("image", cli_kwargs.get("image")) or None,
        "out_dir": pick("out_dir", cli_kwargs.get("out_dir")) or None,
        "class_config_path": pick("class_config_path", cli_kwargs.get("class_config_path")),
        
        #World grid and geometry filter configs
        "extent": dict(
            xmin=pick("xmin", cli_kwargs.get("xmin")) or 0.0,
            ymin=pick("ymin", cli_kwargs.get("ymin")) or 0.0,
            xmax=pick("xmax", cli_kwargs.get("xmax")) or 3500.0,
            ymax=pick("ymax", cli_kwargs.get("ymax")) or 3500.0,
        ),
        "crs": pick("crs", cli_kwargs.get("crs")) or "EPSG:3857",
        "min_area": pick("min_area", cli_kwargs.get("min_area")) or 5.0,
        "min_points": pick("min_points", cli_kwargs.get("min_points")) or 3,
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

    # Load YAML into a typed ExtractConfig, then merge with CLI flags
    tracing_cfg = read_config_file(config, kind="extract")  # type: ignore[arg-type]
    info("override config with CLI args where provided...")
    tracing_cfg.image = image or tracing_cfg.image
    tracing_cfg.out_dir = out_dir or tracing_cfg.out_dir
    
    tracing_cfg.xmin = xmin if xmin is not None else tracing_cfg.xmin
    tracing_cfg.ymin = ymin if ymin is not None else tracing_cfg.ymin
    tracing_cfg.xmax = xmax if xmax is not None else tracing_cfg.xmax
    tracing_cfg.ymax = ymax if ymax is not None else tracing_cfg.ymax
    
    tracing_cfg.crs = crs or tracing_cfg.crs
    tracing_cfg.min_area = min_area if min_area is not None else tracing_cfg.min_area
    tracing_cfg.min_points = min_points if min_points is not None else tracing_cfg.min_points
    tracing_cfg.verbose = verbose if verbose is not None else tracing_cfg.verbose


    info("Resolved configuration:")
    for k, v in tracing_cfg.items():
        setting_config(f"  {k}: {v}")

    # Resolve image and out dir from meta (already merged)
    img = Path(tracing_cfg.image) if tracing_cfg.image else None
    out = Path(tracing_cfg.out_dir) if tracing_cfg.out_dir else None

    if dry_run:
        info("Resolved configuration:")
        for k, v in tracing_cfg.__dict__.items(): setting_config(f"  {k}: {v}")
        raise typer.Exit()

    if not img or not out:
        error("image and out_dir are required (via YAML or flags).")
        raise typer.Exit(code=2)

    out.mkdir(parents=True, exist_ok=True)

    # Run full image -> vector -> raster pipeline
    results = tracing_pipeline.extract_all(image=img, tracing_cfg=tracing_cfg, out_dir=out)
    for k, v in results.items():
        info(f"{k}: {v}")
    success(f"Extraction completed successfully!, Outputs in: {out}")

    logger.teardown()

@app.command("test")
def test():
    success("Test command executed successfully!")

@app.command("extract-image")
def extract_image(
    # I/O
    config: Optional[Path] = typer.Option(
        DEFAULT_CONFIG_FILE_PATH, "--config", "-c",
        help="YAML file with parameters. Flags override YAML.",
        rich_help_panel="I/O"
    ),
):
    """Preprocess a hand-drawn map into outline + filled land mask.

    This is a subset of ``extract-all`` focused on image preprocessing only.
    It writes a centerline outline PNG and a filled land-mask PNG.
    """

    # Load configuration from YAML into ExtractConfig, then normalize
    tracing_cfg = read_config_file(config, kind="extract")  # type: ignore[arg-type]
   

    img = Path(tracing_cfg.image) if tracing_cfg.image else None
    out = Path(tracing_cfg.out_dir) if tracing_cfg.out_dir else None

    if not img or not out:
        error("image and out_dir are required (via YAML for extract-image).")
        raise typer.Exit(code=2)

    out.mkdir(parents=True, exist_ok=True)

    # Preprocess image and write diagnostic PNGs; return value is the land mask
    land_mask = tracing_pipeline.extract_image(image=img, tracing_cfg=tracing_cfg, out_dir=out)
    info(f"Extracted land mask with shape {land_mask.shape} into {out}")

    success("Image extraction completed successfully!")


@app.command("extract-vector")
def extract_vector(
    # I/O
    config: Optional[Path] = typer.Option(
        DEFAULT_CONFIG_FILE_PATH, "--config", "-c",
        help="YAML file with parameters. Flags override YAML.",
        rich_help_panel="I/O",
    ),
    image: Optional[Path] = typer.Option(
        None, "--image", "-i",
        help="Input raster (jpg/png). If omitted, read from YAML.",
        rich_help_panel="I/O",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", "-o",
        help="Output directory. If omitted, read from YAML.",
        rich_help_panel="I/O",
    ),
    xmin: Optional[float] = typer.Option(None, "--xmin", help="Extent min X", rich_help_panel="World grid"),
    ymin: Optional[float] = typer.Option(None, "--ymin", help="Extent min Y", rich_help_panel="World grid"),
    xmax: Optional[float] = typer.Option(None, "--xmax", help="Extent max X", rich_help_panel="World grid"),
    ymax: Optional[float] = typer.Option(None, "--ymax", help="Extent max Y", rich_help_panel="World grid"),
    crs: Optional[str] = typer.Option(None, "--crs", help="Output CRS (e.g., 'EPSG:3857')", rich_help_panel="World grid"),
    min_area: Optional[float] = typer.Option(None, "--min-area", help="Min polygon area (pre-affine)", rich_help_panel="Geometry filters"),
    min_points: Optional[int] = typer.Option(None, "--min-points", help="Min ring vertices", rich_help_panel="Geometry filters"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print resolved config and exit", rich_help_panel="Utility"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging", rich_help_panel="Utility"),
):
    """Extract only vector products (land, waterbody, merged polygons)."""

    log_path = _resolve_log_path()
    logger = Logger(logfile_path=log_path)

    tracing_cfg = read_config_file(config, kind="extract")  # type: ignore[arg-type]
    meta = load_config(
        cfg_obj,
        log_file=log_path,
        verbose=verbose,
        image=image,
        out_dir=out_dir,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        crs=crs,
        min_area=min_area,
        min_points=min_points,
    )

    info("Resolved configuration (extract-vector):")
    for k, v in meta.items():
        setting_config(f"  {k}: {v}")

    img = Path(meta["image"]) if meta["image"] else None
    out = Path(meta["out_dir"]) if meta["out_dir"] else None

    if dry_run:
        logger.teardown()
        raise typer.Exit()

    if not img or not out:
        error("image and out_dir are required (via YAML or flags).")
        logger.teardown()
        raise typer.Exit(code=2)

    out.mkdir(parents=True, exist_ok=True)

    # Run vector extraction; files are written when verbose/out_dir are set
    _even_gdf, _odd_gdf = tracing_pipeline.extract_vectors(img, meta, out_dir=out)

    results = {
        "even_geojson": out / "even.geojson",
        "odd_geojson": out / "odd.geojson",
    }
    for k, v in results.items():
        info(f"{k}: {v}")

    success(f"Vector extraction completed successfully!, Outputs in: {out}")
    logger.teardown()


@app.command("extract-raster")
def extract_raster(
    # I/O
    config: Optional[Path] = typer.Option(
        DEFAULT_CONFIG_FILE_PATH, "--config", "-c",
        help="YAML file with parameters. Flags override YAML.",
        rich_help_panel="I/O",
    ),
    image: Optional[Path] = typer.Option(
        None, "--image", "-i",
        help="Input raster (jpg/png). If omitted, read from YAML.",
        rich_help_panel="I/O",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", "-o",
        help="Output directory. If omitted, read from YAML.",
        rich_help_panel="I/O",
    ),
    xmin: Optional[float] = typer.Option(None, "--xmin", help="Extent min X", rich_help_panel="World grid"),
    ymin: Optional[float] = typer.Option(None, "--ymin", help="Extent min Y", rich_help_panel="World grid"),
    xmax: Optional[float] = typer.Option(None, "--xmax", help="Extent max X", rich_help_panel="World grid"),
    ymax: Optional[float] = typer.Option(None, "--ymax", help="Extent max Y", rich_help_panel="World grid"),
    crs: Optional[str] = typer.Option(None, "--crs", help="Output CRS (e.g., 'EPSG:3857')", rich_help_panel="World grid"),
    min_area: Optional[float] = typer.Option(None, "--min-area", help="Min polygon area (pre-affine)", rich_help_panel="Geometry filters"),
    min_points: Optional[int] = typer.Option(None, "--min-points", help="Min ring vertices", rich_help_panel="Geometry filters"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print resolved config and exit", rich_help_panel="Utility"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging", rich_help_panel="Utility"),
):
    """Extract only class rasters (world/terrain/climate) from the image."""

    log_path = _resolve_log_path()
    logger = Logger(logfile_path=log_path)

    cfg_obj = read_config_file(config, kind="extract")  # type: ignore[arg-type]
    meta = load_config(
        cfg_obj,
        log_file=log_path,
        verbose=verbose,
        image=image,
        out_dir=out_dir,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        crs=crs,
        min_area=min_area,
        min_points=min_points,
    )

    info("Resolved configuration (extract-raster):")
    for k, v in meta.items():
        setting_config(f"  {k}: {v}")

    img = Path(meta["image"]) if meta["image"] else None
    out = Path(meta["out_dir"]) if meta["out_dir"] else None

    if dry_run:
        logger.teardown()
        raise typer.Exit()

    if not img or not out:
        error("image and out_dir are required (via YAML or flags).")
        logger.teardown()
        raise typer.Exit(code=2)

    out.mkdir(parents=True, exist_ok=True)

    # Build class rasters from image path using the tracing pipeline
    results = tracing_pipeline.extract_rasters(img, out, meta)
    for k, v in results.items():
        info(f"{k}: {v}")

    success(f"Raster extraction completed successfully!, Outputs in: {out}")
    logger.teardown()


@app.command("recolor")
def recolor(
    raster: Path = typer.Argument(..., help="Path to class raster (world/terrain/climate)."),
    section: str = typer.Option("terrain", "--section", "-s", help="Section in YAML: base|terrain|climate"),
    config: Path = typer.Option(directories.CONFIG_DIR / "raster_classifications.yml", "--config", "-c", help="YAML file for class colors."),
):
    """Reapply palette from YAML to an existing class raster (in place).
    Example call:
        mapcreator recolor path/to/terrain_class_map.tif --section terrain --config path/to/raster_classifications.yml
    """
    # For recolor we still want a raw YAML dict, not ExtractConfig
    cfg: Dict[str, Any]
    if not config:
        cfg = {}
    else:
        try:
            info(f"Loading config from {config}...")
            with config.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            error(f"(config not found at {config}; using defaults where possible)")
            cfg = {}
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
    # For paint we still want a raw YAML dict, not ExtractConfig
    try:
        info(f"Loading config from {config}...")
        with config.open("r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    except FileNotFoundError:
        error(f"(config not found at {config}; using defaults where possible)")
        cfg = {}
    burn_polygons_into_class_raster(
        raster_path=raster,
        polygons_path=polygons,
        class_config=cfg,
        section=section,
        label_or_id=label,
        output=output,
        overwrite=overwrite,
    )


def main():
    app()

if __name__ == "__main__":
    main()