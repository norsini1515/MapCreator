import pathlib as pl
from mapcreator.globals import configs

BASE_DIR = pl.Path(f"{configs.ROOT_DRIVE}/DnD/WorldBuilding/MapCreator")

# DATA_DIR = pl.Path(f"{configs.ROOT_DRIVE}/DnD/WorldBuilding/map_data/{configs.WORLD_NAME}")
DATA_DIR = pl.Path(f"{configs.ROOT_DRIVE}/DnD/WorldBuilding/MapCreator/data")

IMAGES_DIR = DATA_DIR / 'images'
SHAPEFILES_DIR = DATA_DIR / "shapefiles"
TEMP_FILES_DIR = DATA_DIR / "temp_files"

if __name__ == '__main__':
    print(DATA_DIR)