from mapcreator.scripts import extract_image_from_file, prepare_image
from mapcreator import directories

landmass_base_map = directories.IMAGES_DIR / "landamass_base.jpg"
img = extract_image_from_file(landmass_base_map)
img = prepare_image(img, contrast_factor=2.0)

img.show()
