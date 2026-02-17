"""
Docstring for mapcreator.features.overlays.qt_test

The entrace point for testing Qt integration. 

Loads a simple UI with a QGraphicsView and a menu action to add raster layers.
Working on this to be the basis for working on bringing life into the map.
overlays development, but also to test that the basic Qt setup is working correctly before we build more complex features on top of it.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsView,
)

from mapcreator import (directories as _dirs,
                        config_models
    )
from mapcreator.features.overlays.layer_view import LayerView

# sys.exit()  # temporary to prevent accidental execution while developing

class Loader(QUiLoader):
    def createWidget(self, className, parent=None, name=""):
        # MUST match the "Promoted class name" in Designer exactly
        if className == "LayerView":
            w = LayerView(parent)
            w.setObjectName(name)
            return w
        return super().createWidget(className, parent, name)
    
def new_raster_layer(window, scene: QGraphicsScene, layers_list) -> None:
    """
    Opens a file dialog, loads an image, adds it as a QGraphicsPixmapItem,
    and adds a row to the layers list.
    """
    # get path from user
    path_str, _ = QFileDialog.getOpenFileName(
        window,
        "Open Raster",
        _dirs.RASTER_DATA_DIR.as_posix(),
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
    )
    if not path_str:
        return

    path = Path(path_str)
    name = path.stem

    img = QImage(str(path))
    if img.isNull():
        # If this triggers for your .tif, tell me — we’ll swap in rasterio loading.
        window.statusbar.showMessage(f"Failed to load image: {path.name}", 5000)
        return

    pix = QPixmap.fromImage(img)
    item = QGraphicsPixmapItem(pix)

    # Put new layers on top
    item.setZValue(scene.items().__len__())

    scene.addItem(item)
    layers_list.addItem(name)
    layers_list.setCurrentRow(layers_list.count() - 1)

    # Fit to view if first layer; otherwise leave zoom as-is
    if scene.items().__len__() == 1:
        view = window.findChild(QGraphicsView, "LayerView")
        if view is not None:
            view.fitInView(item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    window.statusbar.showMessage(f"Loaded raster layer: {path.name}", 3000)


if __name__ == "__main__":
    app = QApplication([])

    loader = Loader()
    ui_file = QFile(str(_dirs.UI_LAYOUTS_DIR / "main_app.ui"))
    ui_file.open(QFile.ReadOnly)
    window = loader.load(ui_file)
    ui_file.close()

    class_reg = config_models.read_config_file(None, kind="class_registry")



    # --- find widgets by objectName (must match Designer) ---
    # map_view = window.findChild(QGraphicsView, "mapView")
    map_view = window.findChild(LayerView, "LayerView")
    print(f"map_view: {type(map_view)}")

    layers_list = window.findChild(type(window.layersList), "layersList") if hasattr(window, "layersList") else None
    if layers_list is None:
        print("Falling back to QListWidget for layers_list")
        # safer fallback
        from PySide6.QtWidgets import QListWidget
        layers_list = window.findChild(QListWidget, "layersList")

    action_new_raster = window.findChild(type(window.actionNewTerrain_Layer), "actionNewTerrain_Layer") if hasattr(window, "actionNewTerrain_Layer") else None
    if action_new_raster is None:
        from PySide6.QtGui import QAction
        action_new_raster = window.findChild(QAction, "actionNewTerrain_Layer")

    # if map_view is None or layers_list is None or action_new_raster is None:
    #     raise RuntimeError(
    #         "Could not find one of: mapView, layersList, actionNewTerrain_Layer. "
    #         "Check objectName values in Qt Designer."
    #     )

    # --- scene setup ---
    scene = QGraphicsScene(window)
    map_view.setScene(scene)

    # --- connect menu action ---
    action_new_raster.triggered.connect(
        lambda: new_raster_layer(window, scene, layers_list)
    )

    window.show()
    app.exec()
