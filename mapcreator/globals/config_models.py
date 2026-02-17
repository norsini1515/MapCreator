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
from typing import Dict, Any, Literal, Mapping, overload
from pathlib import Path

import yaml

from mapcreator.globals.logutil import info, process_step, success, warn, error
from mapcreator.globals import directories, configs

Parity = Literal["odd", "even", "none"]

# ID used when falling back to a section's default registry entry
DEFAULT_CLASS_ID = -1

# --- Dataclasses ---------------------------------------------------------
@dataclass(frozen=True)
class ClassDef:
    """Definition of a single class ID within a section."""
    name: str
    color: str
    role: str | None = None #apples to palletete. Used for overlaying
    strength: float | None = None #applies to palette. Used for overlaying
    smooth_px: int | None = None #applies to palette. Used for overlaying

@dataclass(frozen=True)
class ClassRegistry:
    # section -> (id -> ClassDef)
    # "default" entries in the YAML are stored under id DEFAULT_CLASS_ID (-1).
    defines: Mapping[str, Mapping[int, ClassDef]] = field(default_factory=dict)

    def get(self, section: str, class_id: int) -> ClassDef:
        try:
            return self.defines[section][class_id]
        except KeyError as exc:
            available_sections = sorted(self.defines)
            available_ids = sorted(self.defines.get(section, {}).keys())
            raise KeyError(
                f"Missing class definition for section='{section}', id={class_id}. "
                f"Sections={available_sections}. IDs in section={available_ids}"
            ) from exc
    def get_roles(self, section: str, role:str|None=None) -> Dict[int, str]:
        """Return a mapping of class_id to role for all classes in the given section."""
        if role is None:
            try:
                return {class_id: str(class_def.role) for class_id, class_def in self.defines[section].items()}
            except KeyError as exc:
                available_sections = sorted(self.defines)
                raise KeyError(
                    f"Missing section='{section}' when getting roles. "
                    f"Available sections={available_sections}."
                ) from exc
        else:
            role_defs = {}
            for class_id, class_def in self.defines.get(section, {}).items():
                if class_def.role == role:
                    role_defs[class_id] = class_def
            if not role_defs:
                available_roles = set(
                    class_def.role for class_def in self.defines.get(section, {}).values()
                )
                raise KeyError(
                    f"No classes with role='{role}' found in section='{section}'. "
                    f"Available roles in section={available_roles}."
                )
            return role_defs
@dataclass(frozen=True)
class RunSchemeConfig:
    # section -> {"odd": id, "even": id}
    run_scheme: Mapping[str, Mapping[Parity, int]] = field(default_factory=dict)

    def pick_id(self, section: str, parity: Parity) -> int:
        """Return the class id for a section/parity, with fallback.

        Behavior:
        - If the section exists and has an explicit mapping for ``parity``,
          return that id.
        - If the section exists but does not define this ``parity``, return
          ``DEFAULT_CLASS_ID`` so callers can resolve the section's default
          definition from the class registry.
        - If the section itself is missing, raise ``KeyError`` as before.
        """
        section_mapping = self.run_scheme.get(section)
        if section_mapping is None:
            raise KeyError(
                f"Missing run scheme section='{section}'. "
                f"Sections={list(self.run_scheme)}"
            )

        if parity in section_mapping:
            return int(section_mapping[parity])

        # Fallback: let callers use the section's default definition in the
        # registry, which is stored under DEFAULT_CLASS_ID (-1).
        warn(
            f"Missing run scheme mapping for section='{section}', parity='{parity}'. "
            f"Falling back to DEFAULT_CLASS_ID={DEFAULT_CLASS_ID}."
        )
        return DEFAULT_CLASS_ID
    def get_sections(self) -> list[str]:
        return list(self.run_scheme.keys())
    
    def get_keys(self, section: str) -> list[str]:
        try:
            return list(self.run_scheme[section].keys())
        except KeyError as exc:
            raise KeyError(f"Unknown section='{section}'. Sections={list(self.run_scheme)}") from exc
        
@dataclass(frozen=True)
class ClassConfig:
    registry: ClassRegistry
    scheme: RunSchemeConfig

    def resolve(self, *, section: str, parity: Parity) -> tuple[int, ClassDef]:
        class_id = self.scheme.pick_id(section, parity)
        class_def = self.registry.get(section, class_id)
        return class_id, class_def
    
    def get_run_scheme_sections(self) -> list[str]:
        """Return all section names defined in the run scheme config."""
        return list(self.scheme.run_scheme.keys())

    def get_registry_sections(self) -> list[str]:
        """Return all section names defined in the class registry config."""
        return list(self.registry.defines.keys())
    
    def validate(self) -> None:
        """Validate that all IDs in the run scheme exist in the registry."""
        validate_scheme_against_registry(scheme=self.scheme, registry=self.registry)

    def get_even_odd_configs(self) -> tuple[Dict[str, tuple[int, ClassDef]], Dict[str, tuple[int, ClassDef]]]:
        """Return two dicts mapping section names to (id, ClassDef) tuples for even and odd parities."""
        even_cfg: dict[str, tuple[int, ClassDef]] = {}
        odd_cfg: dict[str, tuple[int, ClassDef]] = {}

        for section in self.scheme.get_sections():
            even_cfg[section] = self.resolve(section=section, parity="even")
            odd_cfg[section] = self.resolve(section=section, parity="odd")

        return even_cfg, odd_cfg
    
    def get_roles(self, section: str, role:str|None=None) -> Dict[int, str]:
        """Return a mapping of class_id to role for all classes in the given section."""
        return self.registry.get_roles(section=section, role=role)

@dataclass
class ExtractConfig:
    """Configuration for the image/vector/raster extraction pipeline.

    Mirrors the keys used in ``extract_base_world_configs.yml``.
    CLI flags may still override these values when building the
    runtime ``meta`` dict used by the pipeline.
    """
    #input/output paths
    image: Path | None = None
    out_dir: Path | None = None

    #config paths
    class_run_scheme_configurations_path: Path | None = None
    class_registry_path: Path | None = None

    #image data
    image_shape: tuple | None = None

    #map extent
    xmin: float | None = None
    ymin: float | None = None
    xmax: float | None = None
    ymax: float | None = None
    
    # polygon generation
    crs: str = "EPSG:3857"
    min_points: int | None = configs.MIN_POINTS
    min_area: float | None = configs.MIN_AREA
    compute_extent_polygons: bool = True
    add_parity: bool = True

    #logging
    log_file_name: str | None = None
    verbose: bool | str = False #accepts debug, info
  
# class_id, class_def = class_cfg.resolve(section="terrain", parity="odd")
# class_id -> 0
# class_def.name -> "Waterbody"
# class_def.color -> "#8BBBEB"

# --- YAML loader ---------------------------------------------------------
ConfigKind = Literal[
    "extract",
    "class_registry",
    "class_run_scheme",
]


@overload
def read_config_file(
    path: Path | str | None,
    *,
    kind: Literal["extract"],
) -> ExtractConfig:
    ...

@overload
def read_config_file(
    path: Path | str | None,
    *,
    kind: Literal["class_registry"],
) -> ClassRegistry:
    ...

@overload
def read_config_file(
    path: Path | str | None,
    *,
    kind: Literal["class_run_scheme"],
) -> RunSchemeConfig:
    ...

def resolve_config_path(path: Path | str | None, kind: ConfigKind) -> Path:
    if path is not None:
        p = Path(path) if isinstance(path, str) else path
        if p.exists():
            return p
        else:
            error(f"Provided {kind} Config path does not exist: {p}")
            raise ValueError(f"{kind} Config path does not exist: {p}")
        
    # info(f"{kind=}, {path=}")
    if kind == "extract":
        p = directories.CONFIG_DIR / configs.IMAGE_TRACING_EXTRACT_CONFIGS_FILENAME
        # info(f"Resolved extract config path: {path}")
    elif kind == "class_registry":
        p = directories.CONFIG_DIR / configs.CLASS_REGISTRY_FILENAME
        # info(f"Resolved class registry config path: {path}")
    elif kind == "class_run_scheme":
        p = directories.CONFIG_DIR / configs.CLASS_RUN_SCHEME_CONFIGURATIONS_FILENAME
        # info(f"Resolved class run scheme config path: {path}")
    else:
        error(f"Unsupported config kind: {kind}")
        raise ValueError(f"{kind} Config path does not exist: {p}")

    if not p.exists():
        error(f"{kind} Config path does not exist: {p}")
        raise ValueError(f"{kind} Config path does not exist: {p}")
    return p

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

#--- Config builders ------------------------------------------------------
def build_extract_config(raw: Dict[str, Any]) -> ExtractConfig:
    """Convert raw dict from YAML into :class:`ExtractConfig`."""

    def _to_path(key: str, default: Path | None = None) -> Path | None:
        val = raw.get(key, default)
        if val in (None, ""):
            return None
        return val if isinstance(val, Path) else Path(val)

    extract_cfg = ExtractConfig(
        image=_to_path("image"),
        class_run_scheme_configurations_path=_to_path("class_run_scheme_configurations_path", resolve_config_path(None, kind="class_run_scheme")),
        class_registry_path=_to_path("class_registry_path", resolve_config_path(None, kind="class_registry")),
        out_dir=_to_path("out_dir"),
        xmin=raw.get("xmin"),
        ymin=raw.get("ymin"),
        xmax=raw.get("xmax"),
        ymax=raw.get("ymax"),
        crs=raw.get("crs", "EPSG:3857"),
        min_points=raw.get("min_points"),
        min_area=raw.get("min_area"),
        log_file_name=raw.get("log_file_name"),
        verbose=raw.get("verbose", False),
        compute_extent_polygons=raw.get("compute_extent_polygons", True),
        add_parity=raw.get("add_parity", True),
    )
    # print("verbose:",raw.get("verbose"))
    # print("extract_cfg.verbose:",extract_cfg.verbose)
    # print(bool(extract_cfg.verbose))

    return extract_cfg

def build_run_scheme_config(raw: Dict[str, Any]) -> RunSchemeConfig:
    rs_raw = raw.get("run_scheme") or {}
    run_scheme: Dict[str, Dict[Parity, int]] = {}

    for section, mapping in rs_raw.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"run_scheme['{section}'] must be a mapping.")

        # Only coerce known parities; allow either or both to be present.
        section_mapping: Dict[Parity, int] = {}
        for key in ("odd", "even"):
            if key in mapping:
                section_mapping[key] = int(mapping[key])

        run_scheme[section] = section_mapping

    return RunSchemeConfig(run_scheme=run_scheme)

def build_class_registry(raw: Dict[str, Any]) -> ClassRegistry:
    classes_raw = raw.get("classes") or {}
    defines: Dict[str, Dict[int, ClassDef]] = {}

    for section, section_block in classes_raw.items():
        if not isinstance(section_block, dict):
            raise ValueError(f"classes['{section}'] must be a mapping.")
        defs_raw = section_block.get("defines") or {}
        if not isinstance(defs_raw, dict):
            raise ValueError(f"classes['{section}'].defines must be a mapping of id -> spec.")

        defines[section] = {}
        for class_id, spec in defs_raw.items():
            if not isinstance(spec, dict):
                raise ValueError(f"classes['{section}'].defines['{class_id}'] must be a mapping.")
            if "color" not in spec:
                raise ValueError(f"Missing 'color' for section='{section}', id={class_id}.")
            name = spec.get("name", str(class_id))
            role = spec.get("role", None)
            strength = spec.get("strength", None)
            smooth_px = spec.get("smooth_px", None)

            # Special-case "default" so each section can declare a fallback
            # definition that is used when the run scheme doesn't define a
            # parity-specific id.
            if class_id == "default":
                class_id = DEFAULT_CLASS_ID
            else:
                try:
                    class_id = int(class_id)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid class id '{class_id}' for section='{section}'. "
                        "Expected an integer id or the special key 'default'."
                    ) from exc
                    # continue

            defines[section][class_id] = ClassDef(name=name, color=str(spec["color"]), role=role, strength=strength, smooth_px=smooth_px)

    return ClassRegistry(defines=defines)

def build_class_resolver(
    *,
    registry: ClassRegistry,
    scheme: RunSchemeConfig,
    validate: bool = True,
) -> ClassConfig:
    if validate:
        validate_scheme_against_registry(scheme=scheme, registry=registry)
    return ClassConfig(registry=registry, scheme=scheme)

#--- Config validators ----------------------------------------------------
def validate_scheme_against_registry(*, scheme: RunSchemeConfig, registry: ClassRegistry) -> None:
    for section, mapping in scheme.run_scheme.items():
        if section not in registry.defines:
            raise ValueError(f"run_scheme section '{section}' missing in registry.")
        ids_defined = registry.defines[section].keys()
        missing = [cid for cid in mapping.values() if cid not in ids_defined]
        if missing:
            raise ValueError(f"run_scheme section '{section}' references missing ids: {missing}")

#--- Public API -----------------------------------------------------------        
def read_config_file(
    path: Path | str | None,
    *,
    kind: ConfigKind,
) -> ExtractConfig | ClassRegistry | RunSchemeConfig:
    """Read a YAML config file into a typed dataclass.

    Parameters
    ----------
    path
        Path to a YAML file, or ``None`` to use project defaults
        for the given ``kind``.
    kind
        "extract" for world/image extraction configs (``ExtractConfig``)
        "class_registry" for class registry configs (``ClassRegistry``)
        "class_run_scheme" for class run scheme configs (``RunSchemeConfig``)
    """

    resolved = resolve_config_path(path, kind)
    raw = load_yaml(resolved)
    
    if kind == "extract":
        return build_extract_config(raw)
    
    elif kind == "class_registry":
        reg = build_class_registry(raw)
        info(f"Loaded class registry from {resolved}")
        return reg
    
    elif kind == "class_run_scheme":
        scheme = build_run_scheme_config(raw)
        info(f"Loaded class run scheme from {resolved}")
        return scheme

    # Defensive; Literal typing should prevent this
    raise ValueError(f"Unsupported config kind: {kind}")

def read_class_resolver_from_extract(extract: ExtractConfig) -> ClassConfig:
    registry = read_config_file(extract.class_registry_path, kind="class_registry")
    scheme = read_config_file(extract.class_run_scheme_configurations_path, kind="class_run_scheme")

    return build_class_resolver(registry=registry, scheme=scheme, validate=True)

if __name__ == "__main__":
    """Minimal test harness for config loading.

    - Loads default extract and class configs (using project defaults)
    - Prints a brief summary to verify dataclass construction
    """
    from pprint import pprint
    try:
        info("Testing read_config_file for 'extract'...")
        extract_cfg = read_config_file(None, kind="extract")
        print("-"*100)
        print("ExtractConfig summary:")
        pprint(extract_cfg.__dict__)
        print('-'*100, sep='\n')
    # ------------------------------------------------------------------
        info("Testing ClassConfig construction from ExtractConfig...")
        class_cfg = read_class_resolver_from_extract(extract_cfg)
        success("ClassConfig loaded successfully from ExtractConfig.")
        print("-"*100)
        
        print("ClassConfig summary:")
        pprint(class_cfg.__dict__)
        print('-'*100, sep='\n')

        # process_step("Resolving class for section='base', parity='odd'...")
        # section_id, section_class = class_cfg.resolve(section="base", parity="odd")
        # print(f"{section_id=}, {section_class=}")
        # print('-'*100, sep='\n')

        # process_step("Getting even and odd defs...")
        # even_defs, odd_defs = class_cfg.get_even_odd_configs()
        # print("Even defs:")
        # pprint(even_defs)
        # print("Odd defs:")
        # pprint(odd_defs)
        # print('-'*100, sep='\n')

        process_step("Getting run scheme sections...")
        print(class_cfg.get_run_scheme_sections())
        print('-'*100, sep='\n')

        print(class_cfg.get_roles(section="terrain", role="mountain"))
        # for section_classification in class_cfg.get_run_scheme_sections():
        #     even_id, even_def = even_defs[section_classification]
        #     odd_id, odd_def = odd_defs[section_classification]
        #     info(f"[{section_classification}] even -> {even_id} ({even_def.name})")
        #     info(f"[{section_classification}]  odd -> {odd_id} ({odd_def.name})")
        #     print()

    # ------------------------------------------------------------------
    #     info("Testing read_config_file for 'class_registry'...")
    #     class_registry_cfg = read_config_file(None, kind="class_registry")
    #     print("ClassConfig summary:")
    #     pprint(class_registry_cfg.__dict__)
    #     print('-'*100)
    #      ------------------------------------------------------------------
    #     info("Testing read_config_file for 'class_run_scheme'...")
    #     scheme = read_config_file(None, kind="class_run_scheme")
    #     print("RunSchemeConfig summary:")
    #     pprint(scheme.__dict__)
    #     print('-'*100)
    #     # ------------------------------------------------------------------
        
    #     info("Testing ClassConfig construction...")
    #     resolver = build_class_resolver(registry=class_registry_cfg, scheme=scheme)
    #     # print("ClassConfig summary:")
    #     # print(resolver)

    except Exception as exc:
        error(f"Config models self-test failed: {exc}")

