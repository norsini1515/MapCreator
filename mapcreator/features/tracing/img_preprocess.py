import cv2
import numpy as np
from pathlib import Path

def preprocess_image(img_path: Path, *, contrast_factor=2.0, invert=False, flood_fill=False):
    print(f"Loading image: {img_path}")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    print(f"Original image shape: {img.shape}")

    # Adjust contrast
    img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
    print(f"Applied contrast factor: {contrast_factor}")

    # Threshold using Otsu's method
    thresh_val, bin_ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu's threshold value: {thresh_val}")

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