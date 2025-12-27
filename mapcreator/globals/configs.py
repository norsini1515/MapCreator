
#WORLD CONFIGURATION
#---------------------
WORLD_NAME = 'Atriath'
# WORLD_NAME = 'Testasiia'


#---------------------
#POLYGON GENERATION
#---------------------
MIN_AREA = 2
MIN_POINTS = 3

LAND_DEFS = {"class": "land", "depth": None, "file_name": "land.shp"}
WATERBODY_DEFS = {"class": "waterbody", "depth": None, "file_name": "waterbodies.shp"}

#----------------------



#---------------------
# CONFIGURATION FILES
#---------------------
IMAGE_TRACING_EXTRACT_CONFIGS_NAME = 'extract_base_world_configs.yml'

#--------------------
# Map Export Names
#--------------------
LAND_TRACING_EXTRACT_NAME = "land.shp"
WATER_TRACING_EXTRACT_NAME = "waterbodies.shp"
MERGED_TRACING_EXTRACT_NAME = "merged_base_geography.shp"