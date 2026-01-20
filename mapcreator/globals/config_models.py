"""Typed configuration models and YAML loading helpers.

This module centralizes reading YAML configuration files into
structured dataclasses instead of passing around raw dictionaries.

Currently supported configs
---------------------------
- ExtractConfig: world/image extraction parameters
  (config/extract_base_world_configs.yml)
- ClassConfig: vector/raster class + color configuration
  (config/class_configurations.yml)

The public entry point is :func:`read_config_file`, which knows how
to build the appropriate dataclass for each config type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Literal, Mapping
from pathlib import Path

import yaml

from mapcreator.globals.logutil import info, warn, error
from mapcreator.globals import directories, configs


# --- Dataclasses ---------------------------------------------------------
@dataclass(frozen=True)
class ClassDef:
    """Definition of a single class ID within a section."""
    name: str
    color: str

@dataclass
class ExtractConfig:
    """Configuration for the image/vector/raster extraction pipeline.

    Mirrors the keys used in ``extract_base_world_configs.yml``.
    CLI flags may still override these values when building the
    runtime ``meta`` dict used by the pipeline.
    """

    image: Path | None = None
    class_config_path: Path | None = None
    out_dir: Path | None = None

    #map extent
    xmin: float | None = None
    ymin: float | None = None
    xmax: float | None = None
    ymax: float | None = None
    
    # polygon generation
    crs: str | None = None
    min_points: int | None = None
    min_area: float | None = None

    #logging
    log_file_name: str | None = None
    verbose: bool | None = None
    compute_extent_polygons: bool | None = None

@dataclass
class ClassConfig:
    """Class/label/color configuration for vector + raster products.

    - ``run_scheme``: class_ -> {"even"|"odd" -> int_id}
    - ``class_registry``: class_ -> {int_id -> class_label}
    """
    run_scheme: Dict[str, Dict[str, int]] = field(default_factory=dict)
    registry: Dict[str, Dict[int, ClassDef]] = field(default_factory=dict)

    # Optional convenience: DataFrame view
    def to_df(self):
        import pandas as pd
        rows = []
        for section, id_map in self.registry.items():
            for class_id, ddef in id_map.items():
                rows.append(
                    {
                        "section": section,
                        "id": int(class_id),
                        "name": ddef.name,
                        "color": ddef.color,
                    }
                )
        return pd.DataFrame(rows)

    def resolve(self, section: str, parity: str) -> tuple[int, ClassDef]:
        """Return (class_id, ClassDef) for a given section and parity."""
        try:
            class_id = self.run_scheme[section][parity]
        except KeyError as exc:
            raise KeyError(f"Missing run_scheme mapping for {section=}, {parity=}.") from exc

        try:
            ddef = self.registry[section][class_id]
        except KeyError as exc:
            raise KeyError(f"Missing registry definition for {section=}, {class_id=}.") from exc

        return class_id, ddef
    
# class_id, class_def = class_cfg.resolve(section="terrain", parity="odd")
# class_id -> 0
# class_def.name -> "Waterbody"
# class_def.color -> "#8BBBEB"

# --- YAML loader ---------------------------------------------------------

ConfigKind = Literal["extract", "class"]

def _resolve_path(path: Path | str | None, kind: ConfigKind) -> Path | None:
    """Resolve a config path or fall back to project defaults.

    For ``kind == 'extract'`` this falls back to the standard
    extract config in the project ``config`` directory.

    For ``kind == 'class'`` this falls back to
    ``config/class_configurations.yml``.
    """

    if path is None:
        if kind == "extract":
            return directories.CONFIG_DIR / configs.IMAGE_TRACING_EXTRACT_CONFIGS_FILENAME
        if kind == "class":
            return directories.CONFIG_DIR / configs.CLASS_CONFIGURATIONS_FILENAME
        return None

    if isinstance(path, str):
        return Path(path)
    return path

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file, returning an empty dict on failure with logging."""
    try:
        info(f"Loading config from {path}...")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        info(f"Using config: {path}")
        return dict(data)
    except FileNotFoundError:
        error(f"Config not found at {path}; using defaults where possible.")
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        error(f"Failed to load YAML config at {path}: {exc}")
        return {}

def build_extract_config(raw: Dict[str, Any]) -> ExtractConfig:
    """Convert raw dict from YAML into :class:`ExtractConfig`."""

    def _to_path(key: str) -> Path | None:
        val = raw.get(key)
        if not val:
            return None
        try:
            return Path(val)
        except TypeError:
            return None

    return ExtractConfig(
        image=_to_path("image"),
        class_config_path=_to_path("class_config_path"),
        out_dir=_to_path("out_dir"),
        xmin=raw.get("xmin"),
        ymin=raw.get("ymin"),
        xmax=raw.get("xmax"),
        ymax=raw.get("ymax"),
        crs=raw.get("crs"),
        min_points=raw.get("min_points"),
        min_area=raw.get("min_area"),
        log_file_name=raw.get("log_file_name"),
        verbose=raw.get("verbose"),
        compute_extent_polygons=raw.get("compute_extent_polygons"),
    )

def build_class_config(raw: Dict[str, Any], *, cfg_path: Path | None) -> ClassConfig:
    """Convert raw dict from YAML into :class:`ClassConfig`.
    
        Expects the following structure:
        {
            "run_scheme": {
                "section_name": {
                    "even": int,
                    "odd": int,
                },
                ...
            },
            "registry": {
                "section_name": {
                    int: {
                        "name": str,
                        "color": str,
                    },
                    ...
                },
                ... 
            },
        }

        Raises ValueError on malformed configs.
    
        
    """

    if not raw:
        raise ValueError("Empty class configuration provided.")

    # 1) Pull raw blocks
    run_scheme_raw = raw.get("run_scheme") or {}
    registry_raw = raw.get("registry") or {}

    # 2) Normalize run_scheme: ensure ints
    run_scheme: Dict[str, Dict[str, int]] = {
        section: {parity: int(class_id) for parity, class_id in scheme.items()}
        for section, scheme in run_scheme_raw.items()
    }

    # 3) Normalize registry: ensure int keys + ClassDef values
    registry: Dict[str, Dict[int, ClassDef]] = {}
    for section, id_map in registry_raw.items():
        registry[section] = {
            int(class_id): ClassDef(
                name=spec.get("name", str(class_id)),
                color=spec["color"],
            )
            for class_id, spec in id_map.items()
        }

    # 4) Tiny cross-check: scheme ids exist in registry
    for section, scheme in run_scheme.items():
        if section not in registry:
            raise ValueError(f"run_scheme section '{section}' missing from registry.")
        missing = [cid for cid in scheme.values() if cid not in registry[section]]
        if missing:
            raise ValueError(
                f"run_scheme for section '{section}' references missing ids in registry: {missing}"
            )

    if cfg_path is not None:
        info(f"Loaded class configuration from {cfg_path}")

    return ClassConfig(run_scheme=run_scheme, registry=registry)

def read_config_file(
    path: Path | str | None,
    *,
    kind: ConfigKind,  # must be specified
) -> ExtractConfig | ClassConfig:
    """Read a YAML config file into a typed dataclass.

    Parameters
    ----------
    path
        Path to a YAML file, or ``None`` to use project defaults
        for the given ``kind``.
    kind
        ``"extract"`` for world/image extraction configs
        (``ExtractConfig``), or ``"class"`` for
        vector/raster class configs (``ClassConfig``).
    """

    resolved = _resolve_path(path, kind)
    if resolved is None:
        error(f"No config path could be resolved; returning empty config.")
        raise ValueError("No config path could be resolved.")

    raw = load_yaml(resolved)

    if kind == "extract":
        return build_extract_config(raw)
    if kind == "class":
        return build_class_config(raw, cfg_path=resolved)

    # Defensive; Literal typing should prevent this
    raise ValueError(f"Unsupported config kind: {kind}")

