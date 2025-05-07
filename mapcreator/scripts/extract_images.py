import os
import cv2
import numpy as np
from pathlib import Path
from mapcreator import directories
from PIL import ImageEnhance, Image

'''
Use Pillow to load your map image and apply initial adjustments.
Convert the image into an OpenCV-compatible format (e.g., numpy array).
Use OpenCV for edge detection and contour tracing.
Save results using either library.
'''


def extract_image_from_file(file_path):
    """
    Extract and process an image from a file.

    Args:
        file_path: Path to the image file.

    Returns:
        Processed Pillow image object.
    """
    try:
        img = Image.open(file_path).convert("L")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error loading image: {e}")
        raise
    return img.copy()

def get_image_dimensions(image_path: Path):
    img = extract_image_from_file(image_path)
    return img.width, img.height

def binarize_img(img, threshold=120):
    """
    Binarize an image by converting it to black and white.

    Args:
        img: Pillow image object to binarize.
        threshold: Pixel intensity threshold (0-255). 
                   Pixels above the threshold become white, and pixels below become black.
                   Lower thresholds make the image lighter (more white areas),
                   while higher thresholds make the image darker (more black areas).

    Returns:
        Binarized Pillow image object.
    """
    
    return img.point(lambda p: 0 if p > threshold else 255, mode="1")
    
def prepare_image(img, contrast_factor=2.0):
    """
    Prepare the image by enhancing contrast and binarizing.

    Args:
        img: Pillow image object to prepare.
        contrast_factor: Factor to enhance contrast (default is 2.0).

    Returns:
        Binarized Pillow image object.
    """
    img = img.convert("L") 
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)  # Adjustable contrast factor
    
    binarized = binarize_img(img)
    
    img_array = np.array(binarized).astype(np.uint8) * 255
    
    return invert_image(img_array)

def display_image(image, title="", output=False, resize=False, resize_dim=(1000, 1000), contrast_factor=2.0):
    """
    Display a processed image from a file path or NumPy array.

    Args:
        image (Path | str | np.ndarray): Path to image file or preprocessed NumPy array.
        title (str): Title for display and optional save file.
        output (bool): Whether to save the image.
        resize (bool): Whether to resize for display.
        resize_dim (tuple): Dimensions for resizing (default: (1000, 1000)).
        contrast_factor (float): Only applies when loading from file.
    """
    if isinstance(image, (str, Path)):
        img_path = Path(image)
        img_array = prepare_image(
                        extract_image_from_file(img_path),
                        contrast_factor=contrast_factor
                    )
        
    elif isinstance(image, np.ndarray):
        img_array = image
        
    else:
        raise TypeError("Input must be a NumPy array or path to image file.")

    # # Normalize binary image to uint8
    # if img_array.max() <= 1:
    #     img_array = (img_array * 255).astype(np.uint8)

    display_img = cv2.resize(img_array, resize_dim) if resize else img_array.copy()
    
    print(f'DISPLAYING \"{title.upper()}\"\nclose the window to conitnue.')
    cv2.imshow(title or "Image", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output:
        output_name = title or "processed_image"
        output_path = directories.IMAGES_DIR / f"{output_name}.jpg"
        os.makedirs(output_path.parent, exist_ok=True)
        cv2.imwrite(str(output_path), display_img)
        print(f"✅ Saved debug image to: {output_path}")
    
def image_array_coords(img_array, value=255):
    """
    Extract coordinates of all cells in the image array that are not value (default 255, white).

    Args:
        img_array: NumPy array representing the image.

    Returns:
        List of tuples (x, y), where each tuple is the coordinate of a cell that is not 255.
    """
    # Use NumPy's where function to find indices of non-255 cells
    coords = np.column_stack(np.where(img_array != value))
    
    # Convert to a list of tuples (x, y)
    return coords[:, [1, 0]].astype(int)

def invert_image(img_array):
    return np.abs(255 - img_array)

if __name__ == '__main__':
    # Load the landmass base map
    landmass_base_map = directories.IMAGES_DIR / "old_images/landamass_drawing_base.jpg"
    img = extract_image_from_file(landmass_base_map)
    img_array = prepare_image(img, contrast_factor=2.0)
    
    display_image("Preprocessed Image", img_array)
    
    kernel = np.ones((3, 3), np.uint8)

    # Erode to remove small noise
    eroded_img = cv2.erode(img_array, kernel, iterations=1)
    display_image("Eroded Image", eroded_img)
    
    # Dilate to strengthen edges
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)
    display_image("Dilated Image", dilated_img)
    
    # Closing: Fill small holes
    closed_img = cv2.morphologyEx(dilated_img, cv2.MORPH_CLOSE, kernel)    
    display_image("Closed Image", closed_img)
    
    # Display and process
    print("Image successfully processed and converted to NumPy array.")
