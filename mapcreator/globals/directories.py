import pathlib as pl
from mapcreator.globals import configs

BASE_DIR = pl.Path(f"{configs.ROOT_DRIVE}/MapCreator")

# DATA_DIR = pl.Path(f"{configs.ROOT_DRIVE}/map_data/{configs.WORLD_NAME}")
DATA_DIR = BASE_DIR / "data"

IMAGES_DIR = DATA_DIR / 'images'
SHAPEFILES_DIR = DATA_DIR / "shapefiles"
TEMP_FILES_DIR = DATA_DIR / "temp_files"

if __name__ == '__main__':
    print(f"{BASE_DIR=}")
    print(DATA_DIR)