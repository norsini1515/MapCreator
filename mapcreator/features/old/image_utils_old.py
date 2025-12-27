def preprocess_image(
    image_path,
    contrast_factor: float = 2.0,
    invert: bool = False,
    flood_fill: bool = False,
    display_steps: bool = False
) -> np.ndarray:
    """
    Preprocess an image for contour extraction:
        - Converts to grayscale and binarizes
        - Enhances contrast
        - Optionally inverts
        - Optionally flood-fills background
    
    Args:
        image_path (Path): Path to the input image (.jpg expected).
        contrast_factor (float): Factor for contrast enhancement.
        invert (bool): Whether to invert the binary image.
        flood_fill (bool): Whether to flood-fill the background from (0, 0).
        display_steps (bool): Whether to show intermediate images.
        
    Returns:
        np.ndarray: Preprocessed binary image ready for contour extraction.
    """
    img = extract_images.extract_image_from_file(image_path)
    img_array = extract_images.prepare_image(img, contrast_factor=contrast_factor)
    
    if display_steps:
        extract_images.display_image(img_array, title='After Contrast')

    if invert:
        img_array = invert_image(img_array)
        if display_steps:
            extract_images.display_image(img_array, title='After Inversion')
    
    if flood_fill:
        img_array = flood_fill_img(img_array)
        if display_steps:
            extract_images.display_image(img_array, title='After Flood Fill')

    return img_array