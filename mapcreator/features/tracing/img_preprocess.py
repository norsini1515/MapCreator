import cv2
import numpy as np
from pathlib import Path

from mapcreator.globals.logutil import info, process_step, error, setting_config, success
from mapcreator import directories as _dirs

# -------------------------
# Image preprocessing to binary mask
# -------------------------
def write_image(path: Path | str, image: np.ndarray, *, make_parents: bool = True, log: bool = True, message: str | None = None) -> Path:
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
    ok = cv2.imwrite(str(p), image)
    if not ok:
        raise IOError(f"Failed to write image: {p}")
    if log:
        success(f"{message or 'Wrote image'}: {p}")
    return p

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
            write_image(out_dir / fname, _to_u8_for_display(img), log=False)
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

# --- New helpers: endpoint detection and bridging ---
def _find_endpoints_u8(skel_u8: np.ndarray) -> np.ndarray:
    """Return a uint8 mask of endpoints in a 1px skeleton (8-neighborhood).
    Endpoint has exactly 1 neighbor: 10 (self) + 1 = 11 in the weighted sum trick.
    """
    k = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D((skel_u8 > 0).astype(np.uint8), -1, k)
    return ((nb == 11) & (skel_u8 > 0)).astype(np.uint8)

def _bridge_endpoints(skel: np.ndarray, radius: int = 4, max_links_per_node: int = 1) -> np.ndarray:
    """Greedily connect nearby skeleton endpoints with 1px lines.
    - radius: maximum pixel distance to connect
    - max_links_per_node: how many connections each endpoint may create
    Returns a boolean skeleton with added links.
    """
    if radius <= 0:
        return skel
    u8 = (skel.astype(np.uint8) * 255)
    ep_mask = _find_endpoints_u8(u8)
    ys, xs = np.where(ep_mask > 0)
    if len(xs) < 2:
        return skel

    pts = np.column_stack([xs, ys]).astype(np.int32)
    used = np.zeros(len(pts), dtype=np.int32)

    # Try KDTree for speed; fall back to naive O(N^2).
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(pts)
        for i, p in enumerate(pts):
            if used[i] >= max_links_per_node:
                continue
            idxs = tree.query_ball_point(p, r=float(radius))
            idxs = [j for j in idxs if j != i and used[j] < max_links_per_node]
            if not idxs:
                continue
            d2 = ((pts[idxs] - p)**2).sum(axis=1)
            j = idxs[int(np.argmin(d2))]
            cv2.line(u8, tuple(p), tuple(pts[j]), 255, 1, lineType=cv2.LINE_8)
            used[i] += 1
            used[j] += 1
    except Exception:
        # Naive fallback
        for i in range(len(pts)):
            if used[i] >= max_links_per_node:
                continue
            best_j = -1
            best_d2 = radius*radius + 1
            for j in range(len(pts)):
                if j == i or used[j] >= max_links_per_node:
                    continue
                dx = pts[j,0]-pts[i,0]; dy = pts[j,1]-pts[i,1]
                d2 = dx*dx + dy*dy
                if d2 <= radius*radius and d2 < best_d2:
                    best_d2 = d2; best_j = j
            if best_j >= 0:
                cv2.line(u8, tuple(pts[i]), tuple(pts[best_j]), 255, 1, lineType=cv2.LINE_8)
                used[i] += 1; used[best_j] += 1

    return u8 > 0

def _fill_land_from_outline(
    outline_bool: np.ndarray,
    *,
    dilate_ksize: int = 3,
    dilate_iter: int = 1,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a 1px outline (True == line), produce a filled land mask (True == land).
    Steps:
      - Optionally dilate the outline to seal micro-gaps
      - Compute free space (not barrier)
      - Flood-fill ocean from image border on free space
      - Land = free space minus ocean
    Returns (barrier_used_bool, ocean_bool, land_bool)
    """
    # 1) Ensure barriers are solid enough
    barrier = outline_bool
    if dilate_ksize and dilate_iter and dilate_ksize >= 2 and dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(dilate_ksize), int(dilate_ksize)))
        barrier = cv2.dilate(_bool_to_u8(barrier), k, iterations=int(dilate_iter)) > 0

    _show_step("10a - Barrier For Fill (after dilation)", barrier, verbose, save=True)

    # 2) Free space is everything that's not a barrier
    free_u8 = np.where(barrier, 0, 255).astype(np.uint8)

    # 3) Flood-fill ocean from image borders (multiple seeds)
    h, w = free_u8.shape[:2]
    flood = free_u8.copy()

    def seed_fill(x: int, y: int) -> None:
        if flood[y, x] != 255:
            return
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (int(x), int(y)), 128)

    seeds = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2),
    ]
    for x, y in seeds:
        seed_fill(x, y)

    _show_step("10b - Flooded Ocean (128)", flood, verbose, save=True)

    ocean = flood == 128
    land = flood == 255

    _show_step("10c - Land Mask (filled)", land, verbose, save=True)
    return barrier, ocean, land

def centerline_outline(
    img_path: Path,
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
    pre_dilate_ksize: int = 0,                # 0=disable; try 3
    pre_dilate_iter: int = 0,                 # 0=disable; try 1
    bridge_endpoints_radius: int = 0,         # 0=disable; try 4-6
    bridge_max_links: int = 1,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 1px centerline outline from a hand-drawn raster and return the image data.
    Parameters
    ----------
    img_path : Path
        Path to input image file.
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
    (np.ndarray, np.ndarray)
        Tuple of (outline_u8, skeleton_bool)

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

    # Optional: small dilation to seal hairline gaps before closing
    if isinstance(pre_dilate_ksize, int) and pre_dilate_ksize >= 2 and pre_dilate_iter and pre_dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(pre_dilate_ksize), int(pre_dilate_ksize)))
        mask = cv2.dilate(_bool_to_u8(mask), k, iterations=int(pre_dilate_iter)) > 0
        setting_config(f"Pre-dilate k={pre_dilate_ksize}, iter={pre_dilate_iter}")
        _show_step("05a - After Pre-Dilate", mask, verbose, save=True)
    
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

    # Optional: bridge close endpoints on the skeleton to improve connectivity
    if isinstance(bridge_endpoints_radius, int) and bridge_endpoints_radius > 0:
        skel = _bridge_endpoints(skel, radius=int(bridge_endpoints_radius), max_links_per_node=int(max(1, bridge_max_links)))
        setting_config(f"Bridged endpoints within r={bridge_endpoints_radius}, max_links={bridge_max_links}")
        _show_step("08a - After Bridge Endpoints", skel, verbose, save=True)
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
    if verbose:
        _show_step("10 - Final Outline (computed)", out, verbose, save=True)

    if verbose:
        cv2.destroyAllWindows()

    return out, skel


def fill_outline_mask(
    skel: np.ndarray,
    *,
    fill_dilate_ksize: int = 3,
    fill_dilate_iter: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Fill land from a 1px skeleton outline and return the boolean mask."""
    _, _, land_mask = _fill_land_from_outline(
        skel,
        dilate_ksize=int(fill_dilate_ksize),
        dilate_iter=int(fill_dilate_iter),
        verbose=verbose,
    )
    if verbose:
        _show_step("10d - Final Land (computed)", land_mask, verbose, save=True)
    return land_mask

    

def process_image(
        src_path: Path,
        out_path: Path,
        *,
        # Outline parameters (previously hardcoded defaults)
        contrast: float = 1.0,
        invert_lines: bool = True,
        gaussian_blur_ksize: int = 0,
        close_ksize: int = 0,
        pre_dilate_ksize: int = 1,
        pre_dilate_iter: int = 2,
        bridge_endpoints_radius: int = 8,
        bridge_max_links: int = 1,
        min_stroke_pixels: int = 2,
        prune_spurs: bool = False,
        spur_iterations: int = 6,
        threshold_mode: str = "manual",
        manual_threshold: int | None = 112,
        threshold_offset: int = 0,
        verbose: bool = False,
        # Fill parameters
        fill_dilate_ksize: int = 1,
        fill_dilate_iter: int = 2,
):
    """Process a drawn map image into a centerline outline and filled land.

    Parameters mirror those of `centerline_outline` and `fill_outline_mask` so
    callers can adjust behavior without editing code.
    """
    process_step("Process drawn map image")

    try:
        outline_u8, outline = centerline_outline(
            img_path=src_path,
            contrast=contrast,
            invert_lines=invert_lines,   # typical for pencil-on-paper after Otsu
            gaussian_blur_ksize=gaussian_blur_ksize,
            close_ksize=close_ksize,
            pre_dilate_ksize=pre_dilate_ksize,
            pre_dilate_iter=pre_dilate_iter,
            bridge_endpoints_radius=bridge_endpoints_radius,
            bridge_max_links=bridge_max_links,
            min_stroke_pixels=min_stroke_pixels,
            prune_spurs=prune_spurs,
            spur_iterations=spur_iterations,
            threshold_mode=threshold_mode,
            manual_threshold=manual_threshold,
            threshold_offset=threshold_offset,
            verbose=verbose, # show intermediate steps
        )
        write_image(out_path, outline_u8, message="Wrote centerline outline")
        
    except Exception as e:
        error(f"Centerline export failed: {e}")
        raise

    #now fill land from the produced outline skeleton
    try:
        land_mask = fill_outline_mask(
            outline,
            fill_dilate_ksize=fill_dilate_ksize,
            fill_dilate_iter=fill_dilate_iter,
            verbose=verbose,
        )
        filled_out = out_path.with_name(out_path.stem + "_filled.png")
        print(f"Filled output path: {filled_out}")
        filled_path = write_image(filled_out, _bool_to_u8(land_mask), message="Wrote filled land mask")
        success(f"Wrote filled land mask: {filled_path}")
    except Exception as e:
        error(f"Fill from outline failed: {e}")
        raise

    return land_mask, filled_path

if __name__ == "__main__":
    import os

    src = _dirs.RAW_DATA_DIR / "baselandmass_10282025.jpg"
    outline_png = _dirs.TEST_DATA_DIR / "test_centerline_outline.png"
    filled_png = _dirs.TEST_DATA_DIR / "test_centerline_outline_filled.png"

    info(f"Testing preprocess_image with: {src}")

    land_mask, filled_path = process_image(src, outline_png, verbose=False)
    
    try:
        os.startfile(str(filled_path))
    except Exception as e:
        error(f"Failed to open filled image: {e}")




    