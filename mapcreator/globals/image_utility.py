from mapcreator.globals import configs, directories
from mapcreator.scripts import extract_images

from mapcreator.globals.logutil import info, process_step, error, setting_config, success, warn
from pathlib import Path
import numpy as np
import cv2
import sys

def flood_fill_img(img_array: np.ndarray, seedPoint=(0, 0)) -> np.ndarray:
    """
    Flood fill from top left hand pixel
    Removes the ocean from a binary water image by flood-filling from the edge.

    Returns a new array where only inland lakes/seas are white (255),
    and ocean (connected to edge) is black (0).
    """
    filled = img_array.copy()
    h, w = filled.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Fill from top-left (or any edge pixel known to be ocean)
    cv2.floodFill(filled, mask, seedPoint=seedPoint, newVal=0)

    return filled

def invert_image(img_array):
    return np.abs(255 - img_array)

def write_image(
    path: Path | str,
    image: np.ndarray,
    *,
    make_parents: bool = True,
    log: bool = True,
    message: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Write an image to disk with directory creation and optional logging.

    Parameters
    ----------
    path : Path | str
        Destination path for the image.
    image : np.ndarray
        Image data to write (expects uint8 or a valid OpenCV-encodable array).
    make_parents : bool, optional
        Create parent directories if missing, by default True.
    log : bool, optional
        Emit an info log after write, by default True.
    message : str | None, optional
        Custom message prefix for the log; if None, a generic message is used.

    Returns
    -------
    Path
        The resolved output path written to.
    """
    p = Path(path)
    if make_parents:
        p.parent.mkdir(parents=True, exist_ok=True)

    # Normalize dtype to uint8 so callers can pass bool or float arrays safely
    img_u8 = to_u8_for_display(image)
    if img_u8 is None:
        raise ValueError(f"write_image received None for: {p}")

    # Optionally remove existing file to avoid odd caching behaviors
    if overwrite and p.exists():
        warn(f"Overwriting existing image: {p.resolve()}")
        #delete the existing file
        p.unlink()
        assert not p.exists(), f"unlink failed; file still exists: {p}"

    ok = cv2.imwrite(p, img_u8)

    if not ok or not p.exists():
        raise IOError(f"Failed to write image: {p.resolve()}")
    if log:
        success(f"{message or 'Wrote image'}: {p.resolve()}")
    return p

def is_binary_image(img: np.ndarray, *, tol: int = 3, mid_fraction_max: float = 0.005) -> bool:
    """Heuristically determine if an image is (near-)binary.

    Rules (any satisfied returns True):
    - Unique values are contained within two bands near 0 and 255 (within ``tol``).
    - Fraction of mid-tone pixels (``tol < v < 255-tol``) is less than ``mid_fraction_max``.

    This accepts common "almost binary" cases like {0,1,254,255}.
    """
    if img is None:
        return False
    # Ensure uint8 view for comparisons
    if img.dtype != np.uint8:
        arr = np.clip(img, 0, 255).astype(np.uint8)
    else:
        arr = img

    vals = np.unique(arr)
    info(f"Unique values in image for binary check: {vals}")
    if vals.size == 0:
        return False

    # 1) All unique values fall within low/high bands
    low_band = vals <= tol
    high_band = vals >= (255 - tol)
    if np.all(low_band | high_band):
        return True

    # 2) Allow tiny proportion of mid-tones as noise/anti-aliasing
    mid = (arr > tol) & (arr < (255 - tol))
    mid_frac = float(mid.mean())
    setting_config(f"Near-binary check: tol={tol}, mid-tone fraction={mid_frac:.6f}")
    return mid_frac <= float(mid_fraction_max)

def bool_to_u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)

def u8_to_bool(img: np.ndarray) -> np.ndarray:
    return img > 0

def to_u8_for_display(arr: np.ndarray) -> np.ndarray:
    """Normalize arrays to 0-255 uint8 for cv2.imshow.
    - bool or 0/1 -> 0/255
    - float in [0,1] -> 0/255
    - uint8 left as-is; other integer types clipped to 0..255
    """
    if arr is None:
        return None
    if arr.dtype == np.bool_:
        return (arr.astype(np.uint8) * 255)
    if np.issubdtype(arr.dtype, np.floating):
        a = np.clip(arr, 0.0, 1.0)
        return (a * 255.0).astype(np.uint8)
    if arr.dtype != np.uint8:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def detect_dimensions(img_path: Path | str) -> tuple[int, int]:
    """Detect image dimensions (width, height) from file without loading full image.

    Returns
    -------
    (int, int)
        (width, height) of the image in pixels.

    Raises
    ------
    IOError
        If the image cannot be opened or read.
    """

    # Ensure image dimensions are derived from the actual source image
    try:
        _probe = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if _probe is None:
            raise FileNotFoundError(img_path)
        h, w = _probe.shape[:2]
        h, w = int(h), int(w)
        info(f"Auto-detected image size: {w}x{h}")
    except Exception as e:
        error(f"Failed to detect image dimensions from {img_path}: {e}")
        raise
    
    return (w, h)



    