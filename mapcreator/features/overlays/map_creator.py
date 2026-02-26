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

from pathlib import Path
from typing import Optional, Iterable
from typing import cast

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QIODevice
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QMainWindow,
    QVBoxLayout,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QFormLayout,
    QPushButton,
)

from mapcreator import directories as _dirs
from mapcreator.features.overlays.layer_view import LayerView
from mapcreator.features.overlays.open_layers_list import OpenLayersList, RasterLayer
from mapcreator.features.overlays.palette_controls import PaletteButton, PaletteClass

from mapcreator.globals.config_models import read_config_file
from mapcreator.globals.logutil import info, process_step, error, setting_config, success, warn

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
        if className == "OpenLayersList":
            info(f"[Loader] Creating OpenLayersList widget for {name}")
            w = OpenLayersList(parent)
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

    def __init__(self, class_cfg=None) -> None:
        self.qt_app: QApplication = QApplication([])
        self.window: Optional[QMainWindow] = None

        self.map_view: Optional[LayerView] = None
        self.scene: Optional[QGraphicsScene] = None
        self.open_layers_list: Optional[OpenLayersList] = None
        self.layer_label: Optional[QLabel] = None

        # Actions (grabbed from UI)
        self.action_new_raster: Optional[QAction] = None
        self.action_layer_list: Optional[QAction] = None

        self.layer_control_dock: Optional[QDockWidget] = None
        self.open_layers_dock: Optional[QDockWidget] = None

        self.class_cfg = class_cfg or read_config_file(None, kind="class_configs")

        self._build_ui()
        self._wire_events()

    # -----------------------------
    # UI boot / wiring
    # -----------------------------
    def _build_ui(self) -> None:
        """Load UI, resolve widgets, and initialize the scene."""
        loader = Loader()

        # ui_path = _dirs.UI_LAYOUTS_DIR / "main_app.ui"
        # ui_path = _dirs.UI_LAYOUTS_DIR / "main_app_fixed.ui"
        ui_path = _dirs.UI_LAYOUTS_DIR / "main_app_fixed_layout.ui"
        ui_file = QFile(str(ui_path))
        if not ui_file.exists():
            raise FileNotFoundError(f"UI file not found: {ui_path}")

        if not ui_file.open(QIODevice.OpenModeFlag.ReadOnly):
            raise RuntimeError(f"Failed to open UI file: {ui_path}")

        window = loader.load(ui_file)
        ui_file.close()

        if window is None:
            raise RuntimeError(f"Failed to load UI from: {ui_path}")

        self.window = cast(QMainWindow, window)

        # --- Resolve widgets by objectName ---
        self.map_view = self.window.findChild(LayerView, "LayerView")
        if self.map_view is None:
            raise RuntimeError(
                "Could not find LayerView widget. "
                "Check Qt Designer objectName='LayerView' and promoted class name='LayerView'."
            )

        # Docks + layers list
        self.layer_control_dock = self.window.findChild(QDockWidget, "LayerControl")
        self.open_layers_dock = self.window.findChild(QDockWidget, "OpenLayersListDock")

        self.open_layers_list = self.window.findChild(OpenLayersList, "OpenLayersList")
        if self.open_layers_list is None and self.open_layers_dock is not None:
            self.open_layers_list = self.open_layers_dock.findChild(OpenLayersList, "OpenLayersList")
            if self.open_layers_list is None:
                dock_widget = self.open_layers_dock.widget()
                if isinstance(dock_widget, OpenLayersList):
                    self.open_layers_list = dock_widget

        if self.open_layers_list is None:
            raise RuntimeError(
                "Could not find layers list widget (objectName='OpenLayersList'). "
                "In Qt Designer, create a QDockWidget named 'OpenLayersListDock' and place a promoted "
                "OpenLayersList (QListWidget subclass) inside it named 'OpenLayersList'."
            )

        # Action
        if hasattr(self.window, "actionNew_Layer"):
            self.action_new_raster = getattr(self.window, "actionNew_Layer")
            success("Found actionNew_Layer as attribute of window.")
        else:
            warn("Warning: actionNew_Layer not found as attribute; trying findChild...")
            self.action_new_raster = self.window.findChild(QAction, "actionNew_Layer")

        if self.action_new_raster is None:
            raise RuntimeError("Could not find actionNew_Layer action in UI.")

        if hasattr(self.window, "actionLayer_List"):
            self.action_layer_list = getattr(self.window, "actionLayer_List")
        else:
            self.action_layer_list = self.window.findChild(QAction, "actionLayer_List")

        # --- Scene setup ---
        self.scene = QGraphicsScene(self.window)
        self.map_view.setScene(self.scene)

        # --- Dock placement ---
        # Ensure the layers dock is dockable and placed below the layer-control dock.
        if self.layer_control_dock is not None and self.open_layers_dock is not None:
            if self.layer_control_dock.parent() is not self.window:
                self.layer_control_dock.setParent(self.window)
            if self.open_layers_dock.parent() is not self.window:
                self.open_layers_dock.setParent(self.window)

            # Put both docks on the right; split vertically to stack.
            self.window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.layer_control_dock)
            self.window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.open_layers_dock)
            self.window.splitDockWidget(self.layer_control_dock, self.open_layers_dock, Qt.Orientation.Vertical)

    def _wire_events(self) -> None:
        """Connect UI actions/signals to handlers."""
        assert self.action_new_raster is not None
        assert self.open_layers_list is not None

        self.action_new_raster.triggered.connect(self.new_layer_flow)

        # Toggle for the Layers dock (View -> Dockables -> Layer List)
        if self.action_layer_list is not None and self.open_layers_dock is not None:
            self.action_layer_list.setCheckable(True)
            self.action_layer_list.setChecked(self.open_layers_dock.isVisible())

            self.action_layer_list.toggled.connect(self.open_layers_dock.setVisible)
            self.open_layers_dock.visibilityChanged.connect(self.action_layer_list.setChecked)

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
        assert self.open_layers_list is not None

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
        assert self.open_layers_list is not None

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

        layer = RasterLayer(name=name, path=path, item=item, schema=schema or "unknown")
        self.open_layers_list.add_raster_layer(layer)

        # Fit to view if first layer; otherwise preserve current zoom
        if len(self.open_layers_list.raster_layers) == 1:
            self.map_view.fitInView(item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self._status(f"Loaded raster layer: {path.name}", 3000)

    # -----------------------------
    # Small helpers
    # -----------------------------
    def _status(self, msg: str, timeout_ms: int = 2000) -> None:
        """Write to status bar if available."""
        if self.window is None:
            return

        status_bar = self.window.statusBar()
        if status_bar is not None:
            status_bar.showMessage(msg, timeout_ms)
            return

        # fallback for early dev
        info("[STATUS_PRINT] " + msg)


if __name__ == "__main__":
    # class_cfg = read_config_file(None, kind="class_configs")
    # print(class_cfg.__dict__)
    # schemas = class_cfg.get_run_scheme_sections()
    # print(f"{schemas=}")
    MapCreator().run()
