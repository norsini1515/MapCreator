import os
import cv2
import numpy as np
from PIL import ImageEnhance, Image

from mapcreator import directories

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
    
    img = img.point(lambda p: 0 if p > threshold else 255, mode="1")
    return img
    
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
    
    img = binarize_img(img)
    img = np.array(img).astype(np.uint8) * 255
    
    return img

def display_image(title, img_array, output=False, resize_dim=(1000, 1000)):
    """
    Display and optionally save an image.

    Args:
        title: Title of the window and filename.
        img_array: Image as a NumPy array.
        output: Whether to save the image (default=True).
        resize_dim: Dimensions to resize for display (default=(800, 800)).
    """
    # Normalize binary images to uint8
    if img_array.max() <= 1:
        img_array = (img_array * 255).astype(np.uint8)
    
    # Resize and display
    resized_img = cv2.resize(img_array, resize_dim)
    cv2.imshow(title, resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the image if output is True
    if output:
        output_path = directories.IMAGES_DIR / f"{title}.jpg"
        os.makedirs(output_path.parent, exist_ok=True)
        cv2.imwrite(str(output_path), resized_img)
    
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
    img = prepare_image(img, contrast_factor=2.0)
    
    img_array = np.array(img).astype(np.uint8) * 255  # Convert binary image to uint8
    
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
