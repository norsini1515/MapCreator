# Python Script for Directory Setup
import os

# Base path for the project
base_path = r'F:\\DnD\\WorldBuilding\\MapCreator'

def create_project_structure():
    directories = [
        'mapcreator',
        'mapcreator/globals',
        'mapcreator/data',
        'mapcreator/map',
        'mapcreator/visualization',
        'tests',
        'scripts'
    ]
    files = [
        'mapcreator/__init__.py',
        'mapcreator/globals/__init__.py',
        'mapcreator/globals/constants.py',
        'mapcreator/globals/directories.py',
        'mapcreator/data/__init__.py',
        'mapcreator/data/processing.py',
        'mapcreator/map/__init__.py',
        'mapcreator/map/shapefile.py',
        'mapcreator/map/heightmap.py',
        'mapcreator/visualization/__init__.py',
        'mapcreator/visualization/plots.py',
        'mapcreator/utils.py',
        'tests/__init__.py',
        'tests/test_processing.py',
        'scripts/extract_images.py',
        'scripts/main.py'
    ]

    # Create directories
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)

    # Create empty files
    for file in files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write('')

if __name__ == '__main__':
    create_project_structure()
