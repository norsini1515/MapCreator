import cv2
import numpy as np
from pathlib import Path

from mapcreator.globals.logutil import info, process_step, error, setting_config

# from mapcreator.globals
def preprocess_image(img_path: Path, *, contrast_factor=2.0, invert=False, flood_fill=False, verbose=False) -> np.ndarray:
    """Load and preprocess image to binary mask.
    Steps:
      1. Load as grayscale
      2. Adjust contrast (linear scaling)
      3. Otsu thresholding to binary
      4. Optional inversion
      5. Optional flood fill to close holes in land
      Returns binary mask (np.ndarray, dtype=uint8, values 0/1)
    Parameters
    ----------
    img_path : Path
        Path to input image file.
    contrast_factor : float
        Contrast scaling factor (alpha) for cv2.convertScaleAbs.
    invert : bool
        If True, invert binary image.
    flood_fill : bool
        If True, apply flood fill to close holes in land areas.  
      """
    process_step("Preprocessing image")
    info(f"Loading image: {img_path}")
    
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if verbose:
        cv2.imshow("Original Grayscale Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if img is None:
        raise FileNotFoundError(img_path)
    setting_config(f"Original image shape: {img.shape}")

    # Adjust contrast
    img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
    setting_config(f"Applied contrast factor: {contrast_factor}")

    # Threshold using Otsu's method
    thresh_val, bin_ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    setting_config(f"Otsu's threshold value: {thresh_val}")

    # Invert if specified
    if invert:
        bin_ = 255 - bin_
        print("Inverted binary image")

    # Optional flood fill to close holes
    if flood_fill:
        ff = bin_.copy()
        cv2.floodFill(ff, None, (0,0), 255)
        holes = 255 - ff
        bin_ = cv2.bitwise_or(bin_, holes)
        print("Applied flood fill to close holes")

    unique_vals = np.unique(bin_)
    print(f"Unique values in binary image: {unique_vals}")
    return (bin_ > 0).astype("uint8")

# -------------------------
# Experimental: outline -> centerline export (testing helper)
# -------------------------
def _bool_to_u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)

def _u8_to_bool(img: np.ndarray) -> np.ndarray:
    return img > 0

def _to_u8_for_display(arr: np.ndarray) -> np.ndarray:
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

def _show_step(title: str, img: np.ndarray, verbose: bool, *, save: bool = False, max_w: int = 1280, max_h: int = 900, wait_ms: int = 0) -> None:
    """Show an image scaled to fit on screen using a resizable window and optionally save the step.
    Set wait_ms=0 to wait for a key; >0 waits that many milliseconds.
    If save=True, writes a PNG of the original-sized (non-resized) image to data/processed/test_data.
    """
    if not verbose:
        return
    disp = _to_u8_for_display(img)
    if disp is None:
        return
    # Optional save of original-scale frame
    if save:
        try:
            from mapcreator import directories as _dirs
            import re
            out_dir = _dirs.TEST_DATA_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = re.sub(r"[^A-Za-z0-9._-]+", "_", title.strip()) + ".png"
            cv2.imwrite(str(out_dir / fname), _to_u8_for_display(img))
        except Exception as _e:
            # Non-fatal: continue to display even if save fails
            pass
    h, w = disp.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        disp = cv2.resize(disp, new_size, interpolation=cv2.INTER_AREA)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, disp)
    cv2.waitKey(wait_ms)

def _remove_small_objects_bool(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size from a boolean mask.
    Uses skimage if available; falls back to cv2 connected components.
    """
    try:
        import importlib
        morph = importlib.import_module("skimage.morphology")
        return morph.remove_small_objects(mask, min_size=int(min_size))
    except Exception:
        # Fallback with OpenCV
        u8 = (mask.astype(np.uint8) * 255)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((u8 > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(u8, dtype=np.uint8)
        for i in range(1, num):  # 0 is background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= int(min_size):
                keep[labels == i] = 255
        return keep > 0

def _closing_bool(mask: np.ndarray, ksize: int) -> np.ndarray:
    try:
        import importlib
        morph = importlib.import_module("skimage.morphology")
        # Try to construct a rectangular footprint compatible with different skimage versions
        # print(help(morph.footprint_rectangle))
        try:
            fp = morph.footprint_rectangle((ksize, ksize))  # skimage >= 0.23
        except Exception:
            try:
                fp = morph.footprint_rectangle((ksize, ksize))       # common API
            except Exception as e:
                fp = morph.square(int(ksize))                      # fallback
                error(f"Using square footprint for closing due to error: {e}")
        return morph.closing(mask, fp)
    except Exception:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ksize), int(ksize)))
        return cv2.morphologyEx(_bool_to_u8(mask), cv2.MORPH_CLOSE, k) > 0

def _skeletonize_bool(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a boolean mask to 1px lines.
    Prefers skimage.morphology.skeletonize, falls back to cv2.ximgproc.thinning if available.
    """
    try:
        import importlib
        morph = importlib.import_module("skimage.morphology")
        return morph.skeletonize(mask)
    except Exception:
        # Fallback: OpenCV's ximgproc (requires opencv-contrib)
        try:
            ximg = cv2.ximgproc  # type: ignore[attr-defined]
        except Exception:
            raise ImportError(
                "Skeletonization requires scikit-image or opencv-contrib. Install 'scikit-image' or 'opencv-contrib-python'."
            )
        thin = ximg.thinning((_bool_to_u8(mask)))
        return thin > 0

def export_centerline_outline(
    img_path: Path,
    out_path: Path,
    *,
    contrast: float = 2.0,
    invert_lines: bool = True,
    gaussian_blur_ksize: int = 3,
    close_ksize: int = 3,
    min_stroke_pixels: int = 16,
    prune_spurs: bool = True,
    spur_iterations: int = 6,
    threshold_mode: str = "otsu",            # 'otsu' | 'manual'
    manual_threshold: int | None = None,      # used when threshold_mode='manual'
    threshold_offset: int = 0,                # add/subtract from Otsu when mode='otsu'
    verbose: bool = False,
) -> Path:
    """Create a 1px centerline outline from a hand-drawn raster and write a PNG.
    Parameters
    ----------
    img_path : Path
        Path to input image file.
    out_path : Path
        Path to output PNG file.
    contrast : float
        Contrast scaling factor (alpha) for cv2.convertScaleAbs.
        Set to 1 to leave unchanged.
    invert_lines : bool
        If True, invert binary image after thresholding (typical for pencil drawings).
    gaussian_blur_ksize : int
        Kernel size for optional Gaussian blur (must be odd and >=3). Set to 0 to skip.
    close_ksize : int
        Kernel size for morphological closing. Set to 0 to skip.
    min_stroke_pixels : int
        Minimum stroke size in pixels; smaller objects will be removed. Set to 0 to skip.
    prune_spurs : bool
        If True, prune short spurs from skeleton.
    spur_iterations : int
        Number of iterations to prune spurs.
    verbose : bool
        If True, display intermediate steps.
    
    Returns
    -------
    Path
        Path to output PNG file.

    Steps:
      - optional blur, contrast boost
      - Otsu threshold (invert_lines makes strokes white)
      - small closing + remove tiny specks
      - skeletonize to centerline
      - prune very short spurs (light)
    """
    process_step("Centerline outline export (experimental)")
    info(f"Reading: {img_path}")

    # Load grayscale image
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)
    _show_step("01 - Grayscale", gray, verbose, save=True)
        
    # Apply Gaussian blur if specified
    if isinstance(gaussian_blur_ksize, int) and gaussian_blur_ksize >= 3 and gaussian_blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (gaussian_blur_ksize, gaussian_blur_ksize), 0)
        setting_config(f"Gaussian blur k={gaussian_blur_ksize}")
        _show_step("02 - After Blur", gray, verbose, save=True)
    
    # Adjust contrast
    if isinstance(contrast, (int, float)) and contrast != 1.0:
        gray = cv2.convertScaleAbs(gray, alpha=float(contrast), beta=0)
        setting_config(f"Contrast alpha={contrast}")
        _show_step("03 - After Contrast", gray, verbose, save=True)
    else:
        setting_config("Contrast: unchanged (1.0)")

    # --- Threshold control: Otsu (default) or manual override ---
    threshold_mode = (threshold_mode or "otsu").lower()
    if threshold_mode == "manual" and manual_threshold is not None:
        t_used = int(np.clip(int(manual_threshold), 0, 255))
        threshold_val, binimg = cv2.threshold(gray, t_used, 255, cv2.THRESH_BINARY)
        setting_config(f"Manual threshold used: {t_used}")
    else:
        t_otsu, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t_used = int(np.clip(int(t_otsu) + int(threshold_offset), 0, 255))
        # If offset provided, reapply with fixed threshold at adjusted value
        if int(threshold_offset) != 0:
            _, binimg = cv2.threshold(gray, t_used, 255, cv2.THRESH_BINARY)
            setting_config(f"Otsu base: {t_otsu}; applied offset {threshold_offset} -> used {t_used}")
        else:
            setting_config(f"Otsu's threshold value: {t_used}")
    _show_step("04 - Threshold (pre-invert)", binimg, verbose, save=True)
    
    if invert_lines:
        binimg = 255 - binimg
        setting_config("Invert lines: True (strokes become foreground)")
        _show_step("05 - Threshold (final polarity)", binimg, verbose, save=True)

    mask = _u8_to_bool(binimg)
    if close_ksize and close_ksize >= 3:
        mask = _closing_bool(mask, int(close_ksize))
        setting_config(f"Closing k={close_ksize}")
        
        _show_step("06 - After Closing", mask, verbose, save=True)

    if min_stroke_pixels and min_stroke_pixels > 0:
        before = int(mask.sum())
        mask = _remove_small_objects_bool(mask, int(min_stroke_pixels))
        after = int(mask.sum())
        setting_config(f"Removed small objects: {max(0, before - after)} pixels")
        
        _show_step("07 - After Remove Small Objects", mask, verbose, save=True)

    skel = _skeletonize_bool(mask)
    _show_step("08 - Skeleton", skel, verbose, save=True)
    # import sys
    # sys.exit(f"Debug exit after skeletonization")
    if prune_spurs:
        # Simple endpoint erosion for a few rounds
        u8 = _bool_to_u8(skel)
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
        for _ in range(int(spur_iterations)):
            nb = cv2.filter2D((u8 > 0).astype(np.uint8), -1, kernel)
            endpoints = ((nb == 11) & (u8 > 0))  # 10 (self) + 1 neighbor
            if not endpoints.any():
                break
            u8[endpoints] = 0
        skel = u8 > 0
        setting_config("Pruned short spurs")
        _show_step("09 - After Spur Prune", skel, verbose, save=True)

    out = _bool_to_u8(skel)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    info(f"Wrote centerline outline: {out_path}")
    if verbose:
        _show_step("10 - Final Output (saved)", out, verbose, save=True)
        cv2.destroyAllWindows()

    return out_path


if __name__ == "__main__":
    # Also export a 1px centerline outline using the same test image
    from mapcreator import directories as _dirs
    import os

    src = _dirs.RAW_DATA_DIR / "baselandmass_10282025.jpg"
    dst = _dirs.TEST_DATA_DIR / "test_centerline_outline.png"

    info(f"Testing preprocess_image with: {src}")

    try:
        outdir = export_centerline_outline(
            img_path=src,
            out_path=dst,
            contrast=1.0,
            invert_lines=True,   # typical for pencil-on-paper after Otsu
            gaussian_blur_ksize=0,
            # close_ksize=3,
            close_ksize=0,
            min_stroke_pixels=3,
            prune_spurs=False,
            threshold_mode="manual",
            manual_threshold=112,
            verbose=True, # show intermediate steps
        )
        print(f"Wrote centerline outline: {dst}")
    except Exception as e:
        print(f"Centerline export failed: {e}")

    try:
        os.startfile(str(outdir))
    except Exception:
        pass