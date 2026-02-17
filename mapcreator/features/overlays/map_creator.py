"""
mapcreator.features.overlays.map_creator

Defines MapCreator: the main application/controller for the MapCreator UI.

Responsibilities:
- Load the Qt Designer UI (main_app.ui) using a custom QUiLoader (to support promoted widgets like LayerView)
- Locate key widgets/actions (LayerView canvas, layers list, menu actions)
- Create and own the QGraphicsScene
- Provide layer-loading helpers (starting with raster layers)
- Serve as the single "main" entry point for the desktop app
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QListWidget,
    QMainWindow,
    QVBoxLayout,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QFormLayout,
)

from mapcreator import directories as _dirs
from mapcreator.features.overlays.layer_view import LayerView
from mapcreator.globals.config_models import read_config_file

@dataclass
class RasterLayer:
    """Simple model so list-widget rows can map to real scene items."""
    name: str
    schema: str
    path: Path
    item: QGraphicsPixmapItem

class NewLayerDialog(QDialog):
    """
    Popup dialog to choose the "layer type"/schema (terrain, climate, etc.).
    """

    def __init__(self, schemas: Iterable[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Layer")
        self.setModal(True)

        self._schemas = list(schemas)
        self._selected: Optional[str] = None

        # ---- UI ----
        layout = QVBoxLayout(self)

        blurb = QLabel("Select a layer type (schema) for the new layer:")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        form = QFormLayout()
        self.schema_combo = QComboBox()
        self.schema_combo.addItems(self._schemas)
        form.addRow("Layer type:", self.schema_combo)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Make it feel good: default selection + Enter to accept
        if self._schemas:
            self.schema_combo.setCurrentIndex(0)

    def _on_accept(self) -> None:
        self._selected = self.schema_combo.currentText().strip() or None
        self.accept()

    @property
    def selected_schema(self) -> Optional[str]:
        return self._selected

    @staticmethod
    def get_schema(schemas: Iterable[str], parent=None) -> Optional[str]:
        """
        Convenience: open dialog and return selected schema or None if cancelled.
        """
        dlg = NewLayerDialog(schemas, parent=parent)
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            return dlg.selected_schema
        return None
    

class Loader(QUiLoader):
    """QUiLoader that instantiates promoted widgets (e.g., LayerView) correctly."""
    def createWidget(self, className, parent=None, name=""):
        # MUST match the "Promoted class name" in Designer exactly
        if className == "LayerView":
            w = LayerView(parent)
            w.setObjectName(name)
            return w
        return super().createWidget(className, parent, name)


class MapCreator:
    """
    Main application/controller for the UI.

    Typical usage:
        app = MapCreator()
        app.run()
    """

    def __init__(self) -> None:
        self.qt_app: QApplication = QApplication([])
        self.window: Optional[QMainWindow] = None

        self.map_view: Optional[LayerView] = None
        self.scene: Optional[QGraphicsScene] = None
        self.layers_list: Optional[QListWidget] = None

        # Actions (grabbed from UI)
        self.action_new_raster: Optional[QAction] = None

        self.class_cfg = read_config_file(None, kind="class_configs")

        # Simple in-memory layer registry (index aligns with layers_list row)
        self.raster_layers: list[RasterLayer] = []

        self._build_ui()
        self._wire_events()

    # -----------------------------
    # UI boot / wiring
    # -----------------------------
    def _build_ui(self) -> None:
        """Load UI, resolve widgets, and initialize the scene."""
        loader = Loader()

        ui_path = _dirs.UI_LAYOUTS_DIR / "main_app.ui"
        ui_file = QFile(str(ui_path))
        if not ui_file.exists():
            raise FileNotFoundError(f"UI file not found: {ui_path}")

        if not ui_file.open(QFile.ReadOnly):
            raise RuntimeError(f"Failed to open UI file: {ui_path}")

        window = loader.load(ui_file)
        ui_file.close()

        if window is None:
            raise RuntimeError(f"Failed to load UI from: {ui_path}")

        self.window = window

        # --- Resolve widgets by objectName ---
        self.map_view = self.window.findChild(LayerView, "LayerView")
        if self.map_view is None:
            raise RuntimeError(
                "Could not find LayerView widget. "
                "Check Qt Designer objectName='LayerView' and promoted class name='LayerView'."
            )

        # Layers list (prefer attribute if Designer created it, else findChild)
        if hasattr(self.window, "layersList"):
            self.layers_list = getattr(self.window, "layersList")
        else:
            self.layers_list = self.window.findChild(QListWidget, "layersList")

        if self.layers_list is None:
            raise RuntimeError("Could not find layers list widget (objectName='layersList').")

        # Action
        if hasattr(self.window, "actionNew_Layer"):
            self.action_new_raster = getattr(self.window, "actionNew_Layer")
            print("Found actionNew_Layer as attribute of window.")
        else:
            print("Warning: actionNew_Layer not found as attribute; trying findChild...")
            self.action_new_raster = self.window.findChild(QAction, "actionNew_Layer")

        if self.action_new_raster is None:
            raise RuntimeError("Could not find actionNew_Layer action in UI.")

        # --- Scene setup ---
        self.scene = QGraphicsScene(self.window)
        self.map_view.setScene(self.scene)

    def _wire_events(self) -> None:
        """Connect UI actions/signals to handlers."""
        assert self.action_new_raster is not None
        assert self.layers_list is not None

        # Optional: click a row to "select" layer item in scene (MVP)
        self.layers_list.currentRowChanged.connect(self._on_layer_selected)

        self.action_new_raster.triggered.connect(self.new_layer_flow)

    # -----------------------------
    # Public run loop
    # -----------------------------
    def run(self) -> None:
        assert self.window is not None
        self.window.show()
        self.qt_app.exec()

    # -----------------------------
    # Layer actions
    # -----------------------------
    def new_layer_flow(self) -> None:
        """
        User clicks New Layer:
          1) choose schema (terrain/climate/etc)
          2) then proceed to create/load the layer
        """
        assert self.window is not None

        # 1) gather schema names
        try:
            schemas = self.class_cfg.get_run_scheme_sections()
        except Exception:
            # fallback if config not loaded or method missing
            schemas = ["base", "terrain", "climate"]

        if not schemas:
            self._status("No schemas found in class config.", 4000)
            return

        # 2) show dialog
        schema = NewLayerDialog.get_schema(schemas, parent=self.window)
        if schema is None:
            return  # user cancelled

        # 3) proceed (for now: call your existing raster load, but now you *know* the schema)
        self._status(f"Creating new '{schema}' layer...", 2000)

        # Example: if your next step is "load raster for that layer"
        # you can pass schema through so you can label it or store it.
        self.add_raster_layer_dialog(schema=schema)

    def add_raster_layer_dialog(self, schema: str | None = None) -> None:
        """Open file dialog and add selected raster as a new layer."""
        assert self.window is not None
        assert self.scene is not None
        assert self.map_view is not None
        assert self.layers_list is not None

        path_str, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open Raster",
            _dirs.RASTER_DATA_DIR.as_posix(),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not path_str:
            return

        self.add_raster_layer(Path(path_str), schema=schema)

    def add_raster_layer(self, path: Path, schema: str | None = None) -> None:
        """Add a raster layer to the scene and register it in the layers list."""
        assert self.window is not None
        assert self.scene is not None
        assert self.map_view is not None
        assert self.layers_list is not None

        name = path.stem

        img = QImage(str(path))
        if img.isNull():
            # You can later replace this with rasterio for GeoTIFFs / big rasters.
            self._status(f"Failed to load image: {path.name}", 5000)
            return

        pix = QPixmap.fromImage(img)
        item = QGraphicsPixmapItem(pix)

        # Put new layers on top
        item.setZValue(len(self.scene.items()))

        self.scene.addItem(item)
        self.layers_list.addItem(name)

        layer = RasterLayer(name=name, path=path, item=item, schema=schema or "unknown")
        self.raster_layers.append(layer)

        self.layers_list.setCurrentRow(self.layers_list.count() - 1)

        # Fit to view if first layer; otherwise preserve current zoom
        if len(self.raster_layers) == 1:
            self.map_view.fitInView(item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self._status(f"Loaded raster layer: {path.name}", 3000)

    # -----------------------------
    # Small helpers
    # -----------------------------
    def _on_layer_selected(self, row: int) -> None:
        """MVP: visually indicate selected layer by changing opacity slightly."""
        if row < 0 or row >= len(self.raster_layers):
            return

        # Reset all
        for lyr in self.raster_layers:
            lyr.item.setOpacity(1.0)

        # Highlight selected
        self.raster_layers[row].item.setOpacity(0.85)

    def _status(self, msg: str, timeout_ms: int = 2000) -> None:
        """Write to status bar if available."""
        if self.window is None:
            return
        if hasattr(self.window, "statusbar") and self.window.statusbar is not None:
            self.window.statusbar.showMessage(msg, timeout_ms)
        else:
            # fallback for early dev
            print(msg)


if __name__ == "__main__":
    class_cfg = read_config_file(None, kind="class_configs")
    # print(class_cfg.__dict__)
    schemas = class_cfg.get_run_scheme_sections()
    print(f"{schemas=}")
    MapCreator().run()
