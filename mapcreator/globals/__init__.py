from . import directories
from . import configs
from . import config_models

from typing import Any, Callable

write_image: Callable[..., Any]

# `image_utility` requires OpenCV (`cv2`). Make it optional so UI-only workflows
# (e.g., overlays) can run without installing the full image stack.
try:
	from . import image_utility
	from .image_utility import write_image as _write_image
	write_image = _write_image
except ModuleNotFoundError as exc:
	image_utility = None  # type: ignore[assignment]

	def _missing_write_image(*args, **kwargs) -> Any:
		raise ModuleNotFoundError(
			"OpenCV is required for write_image() (missing 'cv2'). "
			"Install opencv-python (or opencv-python-headless)."
		) from exc

	write_image = _missing_write_image

from .logutil import Logger, info, process_step, warn, error, success, setting_config
from .gdf_tools import to_gdf, dissolve_class, merge_gdfs
from .exporters import export_gdf, export_gdfs