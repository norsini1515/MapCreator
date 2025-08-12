import numpy as np, cv2

def preprocess_image(img_path, *, contrast_factor=2.0, invert=False, flood_fill=False) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
    # Otsu binarize
    _, bin_ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert: bin_ = 255 - bin_
    if flood_fill:
        ff = bin_.copy()
        cv2.floodFill(ff, None, (0, 0), 255)
        holes = 255 - ff
        bin_ = cv2.bitwise_or(bin_, holes)
    return bin_