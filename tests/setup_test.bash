python -c "import mapcreator; print('mapcreator OK:', mapcreator.__file__)"
python -c "import mapcreator.cli; print('cli OK:', mapcreator.cli.__file__)"
python -c "from mapcreator.cli.__main__ import app; print('app OK')"
python -m mapcreator.cli.__main__ --help
mapc --help