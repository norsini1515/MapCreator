
#WORLD CONFIGURATION
#---------------------
WORLD_NAME = 'Atriath'
# WORLD_NAME = 'Testasiia'


#---------------------
#POLYGON GENERATION
#---------------------
MIN_AREA = 2
MIN_POINTS = 3

#depth filled at polygon generation time, here for placeholding
#layer = ??
LAND_DEFS = {"class": "land"}
WATER_DEFS = {"class": "waterbody"}

LAND_EXPORT_DEFS = {"file_name": "land.geojson"}
WATER_EXPORT_DEFS = {"file_name": "water.geojson"}

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