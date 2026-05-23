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

import math
from pathlib import Path
from typing import Optional, Iterable
from typing import cast

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QIODevice, QPointF, QRect, QSize
from PySide6.QtGui import QAction, QBrush, QColor, QImage, QPainter, QPixmap, QRegion
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QWidget,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QMainWindow,
    QStyle,
    QToolBar,
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
from mapcreator.features.overlays.palette_controls import PaletteControls

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

        self.palette_controls: Optional[PaletteControls] = None

        # Active brush state
        self._active_schema: Optional[str] = None
        self._active_class_id: Optional[int] = None
        self._active_class_color: Optional[str] = None
        self._active_class_name: Optional[str] = None

        # Paint layer state
        self._paint_image: Optional[QImage] = None
        self._paint_item: Optional[QGraphicsPixmapItem] = None
        self._paint_origin: QPointF = QPointF(0, 0)
        self._paint_layer_counts: dict[str, int] = {}

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

        self.palette_controls = PaletteControls(main_window=self.window)

        # Layers-panel toolbar — lives inside OpenLayersListDock, below the list
        layers_toolbar = QToolBar("Layer Tools")
        layers_toolbar.setMovable(False)
        layers_toolbar.setFloatable(False)
        layers_toolbar.setIconSize(QSize(18, 18))

        style = QApplication.style()
        new_mask_icon = style.standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder)
        self.action_new_layer_mask = layers_toolbar.addAction(new_mask_icon, "")
        self.action_new_layer_mask.setToolTip("New Layer Mask")

        dock_contents = self.window.findChild(QWidget, "dockWidgetContents_Layers")
        if dock_contents is not None and dock_contents.layout() is not None:
            dock_contents.layout().addWidget(layers_toolbar)

    def _wire_events(self) -> None:
        """Connect UI actions/signals to handlers."""
        assert self.action_new_raster is not None
        assert self.open_layers_list is not None

        self.action_new_raster.triggered.connect(self.new_layer_flow)

        self.open_layers_list.layerSchemaChanged.connect(self._on_layer_schema_changed)
        self.open_layers_list.layerSelected.connect(self._sync_paint_target)
        self.open_layers_list.layerRenamed.connect(self._update_apply_reference_layers)
        self.palette_controls.classSelected.connect(self._on_class_selected)
        self.palette_controls.brushCleared.connect(self._on_brush_cleared)
        self.palette_controls.brushSettingsChanged.connect(self.map_view.set_brush)
        self.map_view.paintStroke.connect(self._on_paint_stroke)
        self.action_new_layer_mask.triggered.connect(self._new_layer_mask)

        self.map_view.set_brush(self.palette_controls.brush_size, self.palette_controls.brush_shape)

        # Toggle for the Layers dock (View -> Dockables -> Layer List)
        if self.action_layer_list is not None and self.open_layers_dock is not None:
            self.action_layer_list.setCheckable(True)
            self.action_layer_list.setChecked(self.open_layers_dock.isVisible())

            self.action_layer_list.toggled.connect(self.open_layers_dock.setVisible)
            self.open_layers_dock.visibilityChanged.connect(self.action_layer_list.setChecked)

    def _on_layer_schema_changed(self, schema: str) -> None:
        if schema == self._active_schema:
            return  # same schema — palette already correct, don't reset brush
        self._active_schema = schema
        self._on_brush_cleared()

        if self.palette_controls is None:
            return
        defines = self.class_cfg.registry.defines.get(schema, {})
        raw = {class_id: {"name": cd.name, "color": cd.color} for class_id, cd in defines.items()}
        self.palette_controls.set_palette_defines(raw)
        self.palette_controls.set_schema_label(schema)

    def _sync_paint_target(self, row: int) -> None:
        """Point _paint_image/_paint_item at the currently selected layer."""
        layers = self.open_layers_list.raster_layers
        if row < 0 or row >= len(layers):
            self._paint_image = None
            self._paint_item = None
            return
        item = layers[row].item
        self._paint_item = item
        self._paint_origin = item.pos()
        self._paint_image = item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)

    def _on_class_selected(self, class_id: int) -> None:
        if self._active_schema is None:
            return
        try:
            class_def = self.class_cfg.registry.get(self._active_schema, class_id)
        except KeyError:
            return
        self._active_class_id = class_id
        self._active_class_color = class_def.color
        self._active_class_name = class_def.name
        assert self.map_view is not None
        self.map_view.set_paint_mode(True)

    def _on_brush_cleared(self) -> None:
        self._active_class_id = None
        self._active_class_color = None
        self._active_class_name = None
        if self.map_view is not None:
            is_eraser = self.palette_controls is not None and self.palette_controls.is_eraser
            if not is_eraser:
                self.map_view.set_paint_mode(False)

    def _build_apply_clip(self, cx: int, cy: int, r: int, shape: str) -> QRegion | None:
        """
        Return a QRegion covering only the pixels in the brush footprint that are NOT
        protected by the Apply filter.  Returns None when no filter is active (paint freely).
        """
        if self.palette_controls is None or self.open_layers_list is None:
            return None
        disabled = self.palette_controls.get_disabled_classes()
        if not disabled:
            return None
        ref_name = self.palette_controls.get_reference_layer_name()
        if not ref_name:
            return None
        ref_layer = next(
            (l for l in self.open_layers_list.raster_layers if l.layer_name == ref_name),
            None,
        )
        if ref_layer is None:
            return None

        ref_img = ref_layer.item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
        ref_origin = ref_layer.item.pos()
        # Offset from paint-image coords → ref-image coords
        ref_dx = int(self._paint_origin.x() - ref_origin.x())
        ref_dy = int(self._paint_origin.y() - ref_origin.y())
        ref_w, ref_h = ref_img.width(), ref_img.height()

        # Pre-build list of protected (r, g, b) tuples for fast comparison
        protected: list[tuple[int, int, int]] = []
        defines = self.class_cfg.registry.defines.get(ref_layer.schema, {})
        for class_id, class_def in defines.items():
            if int(class_id) in disabled:
                c = QColor(class_def.color)
                protected.append((c.red(), c.green(), c.blue()))
        if not protected:
            return None

        TOL = 12
        region = QRegion()

        for dy in range(-r, r + 1):
            py = cy + dy
            x_half = int(math.sqrt(max(0.0, r * r - dy * dy))) if shape != "Square" else r

            run_start: int | None = None
            # +2 sentinel forces any open run to close at the right edge
            for dx in range(-x_half, x_half + 2):
                px = cx + dx
                is_sentinel = dx == x_half + 1

                if is_sentinel:
                    allowed = False
                else:
                    rx, ry = px + ref_dx, py + ref_dy
                    if rx < 0 or ry < 0 or rx >= ref_w or ry >= ref_h:
                        allowed = True
                    else:
                        s = QColor(ref_img.pixel(rx, ry))
                        if s.alpha() < 64:
                            allowed = True  # transparent pixel → no class color here
                        else:
                            sr, sg, sb = s.red(), s.green(), s.blue()
                            allowed = not any(
                                abs(sr - pr) < TOL and abs(sg - pg) < TOL and abs(sb - pb) < TOL
                                for pr, pg, pb in protected
                            )

                if allowed and run_start is None:
                    run_start = px
                elif not allowed and run_start is not None:
                    region = region.united(QRegion(QRect(run_start, py, px - run_start, 1)))
                    run_start = None

        return region  # may be empty (all blocked) — caller must handle

    def _update_apply_reference_layers(self) -> None:
        if self.palette_controls is None or self.open_layers_list is None:
            return
        names = [l.layer_name for l in self.open_layers_list.raster_layers]
        self.palette_controls.update_reference_layers(names)

    def _on_paint_stroke(self, pos: QPointF) -> None:
        if self._paint_image is None:
            return

        is_eraser = self.palette_controls is not None and self.palette_controls.is_eraser
        if not is_eraser and self._active_class_color is None:
            return

        r = self.palette_controls.brush_size if self.palette_controls else 8
        shape = self.palette_controls.brush_shape if self.palette_controls else "Circle"

        img_x = int(pos.x() - self._paint_origin.x())
        img_y = int(pos.y() - self._paint_origin.y())

        clip = self._build_apply_clip(img_x, img_y, r, shape)

        painter = QPainter(self._paint_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        if clip is not None:
            if clip.isEmpty():
                return  # entire brush footprint is protected
            painter.setClipRegion(clip)

        if is_eraser:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.setBrush(QBrush(QColor(0, 0, 0, 255)))
        else:
            painter.setBrush(QBrush(QColor(self._active_class_color)))

        if shape == "Square":
            painter.drawRect(img_x - r, img_y - r, r * 2, r * 2)
        else:
            painter.drawEllipse(img_x - r, img_y - r, r * 2, r * 2)

        painter.end()

        assert self._paint_item is not None
        self._paint_item.setPixmap(QPixmap.fromImage(self._paint_image))

    def _new_layer_mask(self) -> None:
        if self.scene is None or self.map_view is None or self.open_layers_list is None:
            return

        schema = self._active_schema or "unknown"
        name_base = self._active_class_name or self._active_schema or "Paint"
        count = self._paint_layer_counts.get(name_base, 0) + 1
        self._paint_layer_counts[name_base] = count

        existing = self.scene.itemsBoundingRect()
        if existing.isNull():
            vp = self.map_view.viewport()
            w, h = vp.width(), vp.height()
            origin = self.map_view.mapToScene(0, 0)
        else:
            w, h = int(existing.width()), int(existing.height())
            origin = existing.topLeft()

        blank = QImage(w, h, QImage.Format.Format_ARGB32)
        blank.fill(Qt.GlobalColor.transparent)

        item = QGraphicsPixmapItem(QPixmap.fromImage(blank))
        item.setPos(origin)
        item.setZValue(999)
        self.scene.addItem(item)

        layer_name = f"New {name_base} Layer {count}"
        layer = RasterLayer(
            layer_name=layer_name,
            schema=schema,
            path=Path(layer_name),
            item=item,
        )
        # add_raster_layer auto-selects the new row, which fires layerSelected →
        # _sync_paint_target, so _paint_image/_paint_item are set for us.
        self.open_layers_list.add_raster_layer(layer)
        self._update_apply_reference_layers()
        self._status(f"Created mask layer: {layer_name}")

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

        layer = RasterLayer(layer_name=name, path=path, item=item, schema=schema or "unknown")
        self.open_layers_list.add_raster_layer(layer)
        self._update_apply_reference_layers()

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
