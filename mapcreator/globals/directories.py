import pathlib as pl
from mapcreator.globals import constants

BASE_DIR = pl.Path("F:/DnD/WorldBuilding/MapCreator")


DATA_DIR = pl.Path(f"F:/DnD/WorldBuilding/map_data/{constants.WORLD_NAME}")

IMAGES_DIR = DATA_DIR / 'images'
SHAPEFILES_DIR = DATA_DIR / "shapefiles"

if __name__ == '__main__':
    print(DATA_DIR)