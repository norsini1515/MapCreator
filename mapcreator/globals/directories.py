import pathlib as pl
from mapcreator.globals import configs

ROOT_DRIVE = 'C:/Users/nicho/Documents/World Building'

BASE_DIR = pl.Path(f"{ROOT_DRIVE}/MapCreator")

# DATA_DIR = pl.Path(f"{ROOT_DRIVE}/map_data/{configs.WORLD_NAME}")

LOGS_DIR = BASE_DIR / "logs"

# CONFIGURATION DIRECTORIES -----------
CONFIG_DIR = BASE_DIR / "config"

# DATA DIRECTORIES ---------------------
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / 'images'
SHAPEFILES_DIR = DATA_DIR / "shapefiles"
TEMP_FILES_DIR = DATA_DIR / "temp_files"

if __name__ == '__main__':
    print(f"{BASE_DIR=}")
    print(f"{DATA_DIR=}")
    print(f"{LOGS_DIR=}")