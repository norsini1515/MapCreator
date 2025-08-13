'''
mapc extract-all
mapc extract-all --config configs/other.yml

'''
from pathlib import Path
import typer
from mapcreator.features.tracing.pipeline import extract_all as _extract_all
import yaml
from typing import Optional, Dict, Any

DEFAULT_CONFIG_PATH = Path(__file__).parent / "extract_world_base_config.yml"

app = typer.Typer(no_args_is_help=True)

def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _pick(cfg, key, cli_val):
    return cli_val if cli_val is not None else cfg.get(key)

@app.command("extract-all")
def extract_all(
    # I/O
    config: Optional[Path] = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", "-c",
        help="YAML file with parameters. Any flags here override the YAML.",
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
    width: Optional[int] = typer.Option(
        None, "--width",
        help="Working grid width in pixels.",
        rich_help_panel="World grid"
    ),
    height: Optional[int] = typer.Option(
        None, "--height",
        help="Working grid height in pixels.",
        rich_help_panel="World grid"
    ),
    xmin: Optional[float] = typer.Option(
        None, "--xmin",
        help="World extent min X.",
        rich_help_panel="World grid"
    ),
    ymin: Optional[float] = typer.Option(
        None, "--ymin",
        help="World extent min Y.",
        rich_help_panel="World grid"
    ),
    xmax: Optional[float] = typer.Option(
        None, "--xmax",
        help="World extent max X.",
        rich_help_panel="World grid"
    ),
    ymax: Optional[float] = typer.Option(
        None, "--ymax",
        help="World extent max Y.",
        rich_help_panel="World grid"
    ),
    crs: Optional[str] = typer.Option(
        None, "--crs",
        help="CRS for outputs (e.g., 'EPSG:3857').",
        rich_help_panel="World grid"
    ),
    # Preprocessing
    invert: Optional[bool] = typer.Option(
        None, "--invert",
        help="Invert after binarize (use if land is dark on light bg).",
        rich_help_panel="Preprocessing"
    ),
    flood_fill: Optional[bool] = typer.Option(
        None, "--flood-fill",
        help="Flood-fill to close open regions.",
        rich_help_panel="Preprocessing"
    ),
    contrast: Optional[float] = typer.Option(
        None, "--contrast",
        help="Contrast factor before thresholding (e.g., 2.0).",
        rich_help_panel="Preprocessing"
    ),
    # Geometry filters
    min_area: Optional[float] = typer.Option(
        None, "--min-area",
        help="Minimum polygon area to keep (in pixel units pre-affine).",
        rich_help_panel="Geometry filters"
    ),
    min_points: Optional[int] = typer.Option(
        None, "--min-points",
        help="Minimum ring vertex count to keep.",
        rich_help_panel="Geometry filters"
    ),
    # Utility
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print resolved configuration and exit.",
        rich_help_panel="Utility"
    ),
):
    cfg = _load_yaml(config if config and config.exists() else None)

    meta = {
        "crs": _pick(cfg, "crs", crs) or "EPSG:3857",
        "invert": _pick(cfg, "invert", invert) or False,
        "flood_fill": _pick(cfg, "flood_fill", flood_fill) or False,
        "contrast": _pick(cfg, "contrast", contrast) or 2.0,
        "min_area": _pick(cfg, "min_area", min_area) or 5.0,
        "min_points": _pick(cfg, "min_points", min_points) or 3,
        "extent": dict(
            xmin=_pick(cfg, "xmin", xmin) or 0.0,
            ymin=_pick(cfg, "ymin", ymin) or 0.0,
            xmax=_pick(cfg, "xmax", xmax) or 3500.0,
            ymax=_pick(cfg, "ymax", ymax) or 3500.0,
        ),
        "width": _pick(cfg, "width", width) or 1000,
        "height": _pick(cfg, "height", height) or 1000,
    }

    img = _pick(cfg, "image", image)
    out = _pick(cfg, "out_dir", out_dir)

    if dry_run:
        typer.echo("Resolved configuration:")
        typer.echo(f"  image: {img}")
        typer.echo(f"  out_dir: {out}")
        for k, v in meta.items():
            typer.echo(f"  {k}: {v}")
        raise typer.Exit()

    if not img or not out:
        typer.secho("image and out_dir are required (via --config or flags).", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    out.mkdir(parents=True, exist_ok=True)
    results = _extract_all(img, out, meta)
    for k, v in results.items():
        typer.echo(f"{k}: {v}")
    typer.secho(f"[SUCCESS] Done. Outputs in: {out}", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()