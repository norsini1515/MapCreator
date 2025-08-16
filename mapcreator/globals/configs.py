
#------------------
# ROOT_DRIVE = 'P:/GitLab/orsin005/miscellaneous'
# ROOT_DRIVE = 'F:/DnD/WorldBuilding'
ROOT_DRIVE = 'C:/Users/nicho/Documents/World Building'


#WORLD CONFIGURATION
#------------------
WORLD_NAME = 'Htrae'
# WORLD_NAME = 'Testasiia'

#------------------
WORKING_WORLD_IMG_DATE = "04232025"
WORKING_WORLD_IMG_VERSION = "1"
WORKING_WORLD_IMG_NAME = f"{WORLD_NAME}_{WORKING_WORLD_IMG_DATE}_{WORKING_WORLD_IMG_VERSION}.jpg"
#------------------
GEOMETRY_METADATA = {
    "land": {
        "type": "land",
        "level": 0,
        "source": WORKING_WORLD_IMG_NAME,
        "invert": False,
        "flood_fill": False
    },
    "lakes": {
        "type": "lakes",
        "level": 0,
        "source": WORKING_WORLD_IMG_NAME,
        "invert": True,
        "flood_fill": True
    },
    "ocean": {
        "type": "open-water",
        "level": 0,
        "source": "",
    }
}

#------------------
#POLYGON GENERATION
#------------------
MIN_AREA = 5
MIN_POINTS = 4