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

import json
import math
import zipfile
from pathlib import Path
from typing import Optional, cast

from mapcreator.features.overlays.undo_stack import UndoStack, UndoCommand

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QBuffer, QByteArray, QFile, Qt, QIODevice, QPointF, QRect, QSize
from PySide6.QtGui import QAction, QBrush, QColor, QImage, QPainter, QPixmap, QRegion, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSizePolicy,
    QStyle,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mapcreator import directories as _dirs
from mapcreator.features.overlays.layer_view import LayerView
from mapcreator.features.overlays.open_layers_list import OpenLayersList, RasterLayer
from mapcreator.features.overlays.palette_controls import PaletteControls

from mapcreator.globals.config_models import read_config_file
from mapcreator.globals.logutil import error

class WorldDataDialog(QDialog):
    """Popup for editing project-level world metadata (world name, etc.)."""

    def __init__(self, world_name: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("World Data")
        self.setModal(True)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._name_edit = QLineEdit(world_name)
        self._name_edit.setPlaceholderText("e.g. Ettrial")
        form.addRow("World Name:", self._name_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._world_name = world_name
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _on_accept(self) -> None:
        self._world_name = self._name_edit.text().strip()
        self.accept()

    @property
    def world_name(self) -> str:
        return self._world_name

    @staticmethod
    def get_world_data(world_name: str = "", parent=None) -> Optional[str]:
        """Open dialog and return new world name, or None if cancelled."""
        dlg = WorldDataDialog(world_name, parent=parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.world_name
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

    _BRUSH_CLIP_TOL: int = 12  # colour-distance tolerance for the Apply filter mask

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
        self.action_palette_controls: Optional[QAction] = None
        self.action_new_project: Optional[QAction] = None
        self.action_save_layer: Optional[QAction] = None
        self.action_open_layer: Optional[QAction] = None
        self.action_save_project: Optional[QAction] = None
        self.action_save_project_as: Optional[QAction] = None
        self.action_open_project: Optional[QAction] = None
        self.action_world_data: Optional[QAction] = None
        self.action_new_layer_mask: Optional[QAction] = None
        self.action_open_overlay: Optional[QAction] = None

        self._world_name: str = ""
        self.title_label: Optional[QLabel] = None

        self._current_project_path: Optional[Path] = None

        self.layer_control_dock: Optional[QDockWidget] = None
        self.open_layers_dock: Optional[QDockWidget] = None

        self.palette_controls: Optional[PaletteControls] = None

        # Canvas toolbar widgets
        self._top_options_bar: Optional[QFrame] = None
        self._grid_size_combo: Optional[QComboBox] = None
        self._grid_toggle_btn: Optional[QToolButton] = None
        self._zoom_label: Optional[QLabel] = None
        self._coord_label: Optional[QLabel] = None

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

        # Undo/redo
        self.undo_stack: UndoStack = UndoStack()
        self._stroke_start_image: Optional[QImage] = None   # snapshot before current stroke
        self._move_start_pos: Optional[QPointF] = None      # item pos before current drag
        self._resize_start_state: Optional[tuple] = None    # (pos, transform) before resize

        self.class_cfg = class_cfg or read_config_file(None, kind="class_configs")

        self._build_ui()
        self._wire_events()

    # -----------------------------
    # UI boot / wiring
    # -----------------------------
    def _build_ui(self) -> None:
        """Load UI, resolve widgets, and initialize the scene."""
        loader = Loader()

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
        self.layer_control_dock = self.window.findChild(QDockWidget, "layer_palette_controls_dock")
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
        self.action_new_raster = self.window.findChild(QAction, "actionNew_Layer")
        if self.action_new_raster is None:
            raise RuntimeError("Could not find actionNew_Layer action in UI.")

        self.action_layer_list = self.window.findChild(QAction, "actionLayer_List")

        self.action_palette_controls = self.window.findChild(QAction, "actionPalette_Controls")
        self.action_new_project = self.window.findChild(QAction, "actionNew_Project")
        self.action_world_data = self.window.findChild(QAction, "actionWorld_Data")
        self.title_label = self.window.findChild(QLabel, "layer_view_label")
        self.action_save_layer = self.window.findChild(QAction, "actionSave_Layer")
        self.action_open_layer = self.window.findChild(QAction, "actionOpen_Layer")
        self.action_save_project = self.window.findChild(QAction, "actionSave_Project")
        self.action_save_project_as = self.window.findChild(QAction, "actionSave_Project_As")
        self.action_open_project = self.window.findChild(QAction, "actionOpen_Project")

        # --- Scene setup ---
        self.scene = QGraphicsScene(self.window)
        self.scene.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.map_view.setScene(self.scene)

        # --- Canvas frame (margin + toolbars around the map view) ---
        self._build_canvas_frame()

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

        layers_toolbar.addSeparator()

        overlay_icon = style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self.action_open_overlay = layers_toolbar.addAction(overlay_icon, "")
        self.action_open_overlay.setToolTip("Open Overlay (reference image)")

        dock_contents = self.window.findChild(QWidget, "dockWidgetContents_Layers")
        if dock_contents is not None and dock_contents.layout() is not None:
            dock_contents.layout().addWidget(layers_toolbar)

    def _build_canvas_frame(self) -> None:
        """Wrap the LayerView in a white-margined frame with top options strip and bottom toolbar."""
        assert self.window is not None
        assert self.map_view is not None

        central = self.window.centralWidget()
        central_layout = central.layout()  # QVBoxLayout from .ui

        # Detach LayerView from the central layout (keeps widget alive, reparented below)
        central_layout.removeWidget(self.map_view)

        # --- Outer wrapper ---
        canvas_frame = QFrame(central)
        canvas_frame.setObjectName("canvas_frame")
        canvas_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas_frame.setStyleSheet(
            "QFrame#canvas_frame { background: #ebebeb; border: 1px solid #c8c8c8; border-radius: 3px; }"
        )
        canvas_vbox = QVBoxLayout(canvas_frame)
        canvas_vbox.setContentsMargins(8, 8, 8, 0)
        canvas_vbox.setSpacing(0)

        # --- Top options strip (hidden until a toggle is active) ---
        top_bar = QFrame(canvas_frame)
        top_bar.setFrameShape(QFrame.Shape.NoFrame)
        top_bar.setObjectName("canvas_top_bar")
        top_bar.setStyleSheet(
            "QFrame#canvas_top_bar { background: #f0f0f0; border-bottom: 1px solid #cccccc; }"
        )
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(6, 3, 6, 3)
        top_layout.setSpacing(6)

        grid_size_label = QLabel("Grid size:")
        self._grid_size_combo = QComboBox()
        self._grid_size_combo.addItems(["50", "100", "200"])
        self._grid_size_combo.setCurrentIndex(1)
        self._grid_size_combo.setFixedWidth(64)
        top_layout.addWidget(grid_size_label)
        top_layout.addWidget(self._grid_size_combo)
        top_layout.addStretch()
        top_bar.hide()
        self._top_options_bar = top_bar
        canvas_vbox.addWidget(top_bar)

        # --- Map view fills the middle ---
        canvas_vbox.addWidget(self.map_view, 1)

        # --- Bottom toolbar ---
        bottom_bar = QFrame(canvas_frame)
        bottom_bar.setObjectName("canvas_bottom_bar")
        bottom_bar.setFrameShape(QFrame.Shape.NoFrame)
        bottom_bar.setStyleSheet(
            "QFrame#canvas_bottom_bar { background: #f0f0f0; border-top: 1px solid #cccccc; }"
        )
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(6, 2, 6, 2)
        bottom_layout.setSpacing(8)

        self._grid_toggle_btn = QToolButton()
        self._grid_toggle_btn.setText("Grid")
        self._grid_toggle_btn.setCheckable(True)
        self._grid_toggle_btn.setToolTip("Toggle grid overlay")
        bottom_layout.addWidget(self._grid_toggle_btn)

        def _vsep() -> QFrame:
            s = QFrame()
            s.setFrameShape(QFrame.Shape.VLine)
            s.setFrameShadow(QFrame.Shadow.Sunken)
            return s

        bottom_layout.addWidget(_vsep())

        self._zoom_label = QLabel("100%")
        self._zoom_label.setToolTip("Zoom level")
        self._zoom_label.setMinimumWidth(44)
        bottom_layout.addWidget(self._zoom_label)

        bottom_layout.addWidget(_vsep())

        self._coord_label = QLabel("x: —   y: —")
        self._coord_label.setToolTip("Scene coordinates under cursor")
        bottom_layout.addWidget(self._coord_label)
        bottom_layout.addStretch()
        canvas_vbox.addWidget(bottom_bar)

        # Re-insert canvas_frame into the central layout (after the title label)
        central_layout.addWidget(canvas_frame, 1)

    def _wire_events(self) -> None:
        """Connect UI actions/signals to handlers."""
        assert self.window is not None
        assert self.action_new_raster is not None
        assert self.open_layers_list is not None

        self.action_new_raster.triggered.connect(self.new_layer_flow)

        self.open_layers_list.layerSchemaChanged.connect(self._on_layer_schema_changed)
        self.open_layers_list.layerSelected.connect(self._sync_paint_target)
        self.open_layers_list.duplicateRequested.connect(self._on_duplicate_layer)
        self.open_layers_list.deleteRequested.connect(self._on_delete_layer)
        self.open_layers_list.mergeRequested.connect(self._on_merge_layers)
        self.open_layers_list.layerRenamed.connect(self._on_layer_renamed)
        self.open_layers_list.reorderOccurred.connect(self._on_layer_reordered)
        self.palette_controls.classSelected.connect(self._on_class_selected)
        self.palette_controls.brushCleared.connect(self._on_brush_cleared)
        self.palette_controls.brushSettingsChanged.connect(self.map_view.set_brush)
        self.palette_controls.schemaComboChanged.connect(self._on_layer_schema_changed)
        self.palette_controls.snapToGridChanged.connect(self._on_snap_mode_changed)
        self.palette_controls.moveModeChanged.connect(self._on_move_mode_changed)
        self.map_view.paintStroke.connect(self._on_paint_stroke)
        self.map_view.paintEnded.connect(self._on_paint_ended)
        self.map_view.layerMoved.connect(self._on_layer_moved)
        self.map_view.layerMoveEnded.connect(self._on_layer_move_ended)
        self.map_view.handleDragged.connect(self._on_handle_dragged)
        self.map_view.handleDragEnded.connect(self._on_handle_drag_ended)
        self.action_new_layer_mask.triggered.connect(self._new_layer_mask)
        self.action_open_overlay.triggered.connect(self.open_overlay_file)

        # Undo / redo keyboard shortcuts
        from PySide6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence.StandardKey.Undo, self.window).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self.window).activated.connect(self._redo)

        self.map_view.set_brush(self.palette_controls.brush_size, self.palette_controls.brush_shape)

        # Populate schema combobox from the registry (dynamic — reads YAML each time)
        self._populate_schema_combo()

        # Canvas toolbar
        assert self._grid_toggle_btn is not None
        assert self._grid_size_combo is not None
        assert self._zoom_label is not None
        assert self._coord_label is not None
        self._grid_toggle_btn.toggled.connect(self._on_grid_toggled)
        self._grid_size_combo.currentTextChanged.connect(self._on_grid_size_changed)
        self.map_view.zoomChanged.connect(self._on_zoom_changed)
        self.map_view.cursorMoved.connect(self._on_cursor_moved)
        self.map_view.cursorLeft.connect(lambda: self._coord_label.setText("x: —   y: —"))
        self._on_zoom_changed(self.map_view.transform().m11())

        if self.action_new_project is not None:
            self.action_new_project.triggered.connect(self._new_project_flow)
        if self.action_world_data is not None:
            self.action_world_data.triggered.connect(self._show_world_data_dialog)

        if self.action_save_layer is not None:
            self.action_save_layer.triggered.connect(self.save_layer)
        if self.action_open_layer is not None:
            self.action_open_layer.triggered.connect(self.open_layer_file)
        menu_open_overlay = self.window.findChild(QAction, "actionOpen_Overlay")
        if menu_open_overlay is not None:
            menu_open_overlay.triggered.connect(self.open_overlay_file)
        if self.action_save_project is not None:
            self.action_save_project.triggered.connect(self.save_project)
        if self.action_save_project_as is not None:
            self.action_save_project_as.triggered.connect(self.save_project_as)
        if self.action_open_project is not None:
            self.action_open_project.triggered.connect(self.open_project)

        # Toggle for the Layers dock (View -> Dockables -> Layer List)
        if self.action_layer_list is not None and self.open_layers_dock is not None:
            self.action_layer_list.setCheckable(True)
            self.action_layer_list.setChecked(self.open_layers_dock.isVisible())
            self.action_layer_list.toggled.connect(self.open_layers_dock.setVisible)
            self.open_layers_dock.visibilityChanged.connect(self.action_layer_list.setChecked)

        # Toggle for the Palette Controls dock
        if self.action_palette_controls is not None and self.layer_control_dock is not None:
            self.action_palette_controls.setCheckable(True)
            self.action_palette_controls.setChecked(self.layer_control_dock.isVisible())
            self.action_palette_controls.toggled.connect(self.layer_control_dock.setVisible)
            self.layer_control_dock.visibilityChanged.connect(self.action_palette_controls.setChecked)

    def _populate_schema_combo(self) -> None:
        """Reload class_registry.yml and push schema names into the palette combobox."""
        if self.palette_controls is None:
            return
        try:
            self.class_cfg = read_config_file(None, kind="class_configs")
            schemas = self.class_cfg.get_registry_sections()
        except Exception:
            schemas = list(self.class_cfg.get_registry_sections()) if self.class_cfg else []
        if schemas:
            self.palette_controls.populate_schemas(schemas)

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
        # Keep the combobox in sync when schema changes via layer selection
        self.palette_controls.set_schema(schema)

    def _sync_paint_target(self, row: int) -> None:
        """Point _paint_image/_paint_item at the currently selected layer."""
        layers = self.open_layers_list.raster_layers
        if row < 0 or row >= len(layers):
            self._paint_image = None
            self._paint_item = None
            return
        layer = layers[row]
        item = layer.item
        self._paint_item = item
        self._paint_origin = item.pos()
        if layer.is_overlay:
            self._paint_image = None  # overlays are reference only — not paintable
            if self.map_view is not None:
                self.map_view.set_overlay_item(item)
        else:
            self._paint_image = item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
            if self.map_view is not None:
                self.map_view.set_overlay_item(None)

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

        All visible non-paint layers are checked: a pixel is blocked if ANY of them
        contains a protected class color at that position.
        """
        if self.palette_controls is None or self.open_layers_list is None:
            return None
        disabled = self.palette_controls.get_disabled_classes()
        if not disabled:
            return None

        # Collect every visible layer that isn't what we're currently painting on
        ref_layers = [
            layer for layer in self.open_layers_list.raster_layers
            if layer.item.isVisible() and layer.item is not self._paint_item
        ]
        if not ref_layers:
            return None

        # Build the protected-color list from each reference layer's own schema
        # (deduped so the same color from two layers isn't checked twice)
        seen_colors: set[tuple[int, int, int]] = set()
        protected: list[tuple[int, int, int]] = []
        for ref_layer in ref_layers:
            defines = self.class_cfg.registry.defines.get(ref_layer.schema, {})
            for class_id, class_def in defines.items():
                if int(class_id) in disabled:
                    c = QColor(class_def.color)
                    rgb = (c.red(), c.green(), c.blue())
                    if rgb not in seen_colors:
                        seen_colors.add(rgb)
                        protected.append(rgb)
        if not protected:
            return None

        # Pre-cache each reference layer's image and paint→ref pixel offset
        # each entry: (img, off_x, off_y, width, height)
        cache: list[tuple] = []
        for ref_layer in ref_layers:
            img = ref_layer.item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
            ref_origin = ref_layer.item.pos()
            off_x = int(self._paint_origin.x() - ref_origin.x())
            off_y = int(self._paint_origin.y() - ref_origin.y())
            cache.append((img, off_x, off_y, img.width(), img.height()))

        tol = self._BRUSH_CLIP_TOL
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
                    allowed = True
                    for img, off_x, off_y, ref_w, ref_h in cache:
                        rx, ry = px + off_x, py + off_y
                        if rx < 0 or ry < 0 or rx >= ref_w or ry >= ref_h:
                            continue  # outside this layer's bounds — no color here
                        s = QColor(img.pixel(rx, ry))
                        if s.alpha() < 64:
                            continue  # transparent — no class color here
                        sr, sg, sb = s.red(), s.green(), s.blue()
                        if any(
                            abs(sr - pr) < tol and abs(sg - pg) < tol and abs(sb - pb) < tol
                            for pr, pg, pb in protected
                        ):
                            allowed = False
                            break  # one layer blocked it — no need to check the rest

                if allowed and run_start is None:
                    run_start = px
                elif not allowed and run_start is not None:
                    region = region.united(QRegion(QRect(run_start, py, px - run_start, 1)))
                    run_start = None

        return region  # may be empty (all blocked) — caller must handle

    def _on_paint_stroke(self, pos: QPointF) -> None:
        if self._paint_image is None:
            return

        is_eraser = self.palette_controls is not None and self.palette_controls.is_eraser
        if not is_eraser and self._active_class_color is None:
            return

        if self._stroke_start_image is None:
            self._stroke_start_image = self._paint_image.copy()

        if self.palette_controls is not None and self.palette_controls.snap_to_grid:
            self._paint_snap_cell(pos, is_eraser)
            return

        r = self.palette_controls.brush_size if self.palette_controls else 8
        shape = self.palette_controls.brush_shape if self.palette_controls else "Circle"

        img_x = int(pos.x() - self._paint_origin.x())
        img_y = int(pos.y() - self._paint_origin.y())

        clip = self._build_apply_clip(img_x, img_y, r, shape)

        painter = QPainter(self._paint_image)
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

    def _paint_snap_cell(self, pos: QPointF, is_eraser: bool) -> None:
        """Fill the entire grid cell under `pos` on the current paint layer."""
        if self.scene is None or self._paint_image is None or self._paint_item is None:
            return
        bounds = self.scene.itemsBoundingRect()
        if bounds.isNull():
            return

        count = int(self._grid_size_combo.currentText()) if self._grid_size_combo else 100
        cell_w = bounds.width()  / count
        cell_h = bounds.height() / count
        if cell_w < 1 or cell_h < 1:
            return

        col = max(0, min(count - 1, int((pos.x() - bounds.left()) / cell_w)))
        row = max(0, min(count - 1, int((pos.y() - bounds.top())  / cell_h)))

        # Cell rect in paint-image coordinates
        img_x = int(bounds.left() + col * cell_w - self._paint_origin.x())
        img_y = int(bounds.top()  + row * cell_h - self._paint_origin.y())
        img_w = max(1, round(cell_w))
        img_h = max(1, round(cell_h))

        # Reuse the apply filter: treat the cell as a square brush centred on the cell
        cell_cx = img_x + img_w // 2
        cell_cy = img_y + img_h // 2
        r = max(img_w, img_h) // 2 + 1
        clip = self._build_apply_clip(cell_cx, cell_cy, r, "Square")

        painter = QPainter(self._paint_image)
        painter.setPen(Qt.PenStyle.NoPen)

        if clip is not None:
            if clip.isEmpty():
                painter.end()
                return
            painter.setClipRegion(clip)

        if is_eraser:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.setBrush(QBrush(QColor(0, 0, 0, 255)))
        else:
            painter.setBrush(QBrush(QColor(self._active_class_color)))

        painter.drawRect(img_x, img_y, img_w, img_h)
        painter.end()
        self._paint_item.setPixmap(QPixmap.fromImage(self._paint_image))

    def _on_duplicate_layer(self, row: int) -> None:
        if self.scene is None or self.open_layers_list is None:
            return
        layers = self.open_layers_list.raster_layers
        if not (0 <= row < len(layers)):
            return

        src = layers[row]
        new_item = QGraphicsPixmapItem(src.item.pixmap().copy())
        new_item.setPos(src.item.pos())
        new_item.setVisible(src.item.isVisible())
        new_item.setOpacity(src.item.opacity())
        new_item.setZValue(src.item.zValue() + 1)
        self.scene.addItem(new_item)

        layer = RasterLayer(
            layer_name=f"Copy of {src.layer_name}",
            schema=src.schema,
            path=src.path,
            item=new_item,
        )
        self.open_layers_list.add_raster_layer(layer)
        self._push_create_cmd(layer, f"Duplicate '{layer.layer_name}'")
        self._status(f"Duplicated: {layer.layer_name}", 2000)

    def _on_delete_layer(self, row: int) -> None:
        if self.scene is None or self.open_layers_list is None:
            return
        layers = self.open_layers_list.raster_layers
        if not (0 <= row < len(layers)):
            return
        layer = layers[row]
        saved_z = layer.item.zValue()

        if self._paint_item is layer.item:
            self._paint_image = None
            self._paint_item = None
        if layer.is_overlay and self.map_view is not None:
            self.map_view.set_overlay_item(None)
        self.scene.removeItem(layer.item)
        self.open_layers_list.remove_layer(row)

        def undo_delete() -> None:
            assert self.scene is not None
            assert self.open_layers_list is not None
            layer.item.setZValue(saved_z)
            self.scene.addItem(layer.item)
            self.open_layers_list.add_raster_layer(layer)

        def redo_delete() -> None:
            r = self._find_layer_row(layer)
            if r < 0:
                return
            if layer.item is self._paint_item:
                self._paint_image = None
                self._paint_item = None
            if layer.is_overlay and self.map_view is not None:
                self.map_view.set_overlay_item(None)
            assert self.scene is not None
            assert self.open_layers_list is not None
            self.scene.removeItem(layer.item)
            self.open_layers_list.remove_layer(r)

        self.undo_stack.push(UndoCommand(
            undo_fn=undo_delete, redo_fn=redo_delete,
            description=f"Delete '{layer.layer_name}'"
        ))
        self._status(f"Deleted layer: {layer.layer_name}", 2000)

    def _on_merge_layers(self, rows: list) -> None:
        if self.scene is None or self.open_layers_list is None:
            return
        layers = self.open_layers_list.raster_layers
        valid = [r for r in rows if 0 <= r < len(layers)]
        if len(valid) < 2:
            return

        selected = [layers[r] for r in valid]

        if any(l.is_overlay for l in selected):
            self._status("Cannot merge overlay layers.", 3000)
            return

        pre_merge_order = list(self.open_layers_list.raster_layers)

        selected_sorted = sorted(selected, key=lambda l: l.item.zValue())
        top_layer = selected_sorted[-1]

        bounds = self.scene.itemsBoundingRect()
        if bounds.isNull():
            return
        w, h = int(bounds.width()), int(bounds.height())
        origin = bounds.topLeft()

        merged_img = QImage(w, h, QImage.Format.Format_ARGB32)
        merged_img.fill(Qt.GlobalColor.transparent)

        painter = QPainter(merged_img)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        for layer in selected_sorted:
            src = layer.item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
            off_x = int(layer.item.pos().x() - origin.x())
            off_y = int(layer.item.pos().y() - origin.y())
            painter.drawImage(off_x, off_y, src)
        painter.end()

        new_item = QGraphicsPixmapItem(QPixmap.fromImage(merged_img))
        new_item.setPos(origin)
        new_item.setZValue(top_layer.item.zValue())
        self.scene.addItem(new_item)

        new_layer = RasterLayer(
            layer_name=f"Merged: {top_layer.layer_name}",
            schema=top_layer.schema,
            path=Path("merged"),
            item=new_item,
        )

        for row in sorted(valid, reverse=True):
            layer = layers[row]
            if layer.item is self._paint_item:
                self._paint_image = None
                self._paint_item = None
            self.scene.removeItem(layer.item)
            self.open_layers_list.remove_layer(row)

        self.open_layers_list.add_raster_layer(new_layer)
        post_merge_order = list(self.open_layers_list.raster_layers)

        removed = [l for l in pre_merge_order if l not in post_merge_order]
        added   = [l for l in post_merge_order if l not in pre_merge_order]

        def undo_merge() -> None:
            assert self.scene is not None
            assert self.open_layers_list is not None
            for l in added:
                r = self._find_layer_row(l)
                if r >= 0:
                    if l.item is self._paint_item:
                        self._paint_image = None
                        self._paint_item = None
                    self.scene.removeItem(l.item)
            for l in removed:
                self.scene.addItem(l.item)
            self.open_layers_list.restore_order(pre_merge_order)

        def redo_merge() -> None:
            assert self.scene is not None
            assert self.open_layers_list is not None
            for l in removed:
                r = self._find_layer_row(l)
                if r >= 0:
                    if l.item is self._paint_item:
                        self._paint_image = None
                        self._paint_item = None
                    self.scene.removeItem(l.item)
            for l in added:
                self.scene.addItem(l.item)
            self.open_layers_list.restore_order(post_merge_order)

        self.undo_stack.push(UndoCommand(
            undo_fn=undo_merge, redo_fn=redo_merge,
            description=f"Merge {len(valid)} layers"
        ))
        self._status(f"Merged {len(valid)} layers → {new_layer.layer_name}", 3000)

    def _new_layer_mask(self) -> None:
        if self.scene is None or self.map_view is None or self.open_layers_list is None:
            return

        # Schema comes from the palette combobox, falling back to the active schema
        schema = (
            (self.palette_controls.current_schema if self.palette_controls else None)
            or self._active_schema
            or "unknown"
        )
        name_base = self._active_class_name or schema or "Paint"
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
        self._push_create_cmd(layer, f"New layer '{layer_name}'")
        self._status(f"Created mask layer: {layer_name}")

    # -----------------------------
    # Save / open helpers
    # -----------------------------
    def _image_to_tif_bytes(self, image: QImage) -> bytes:
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buf, "TIFF")
        buf.close()
        return bytes(buf.data())

    def _tif_bytes_to_image(self, data: bytes) -> QImage:
        img = QImage()
        img.loadFromData(QByteArray(data), "TIFF")
        return img

    def _layer_to_meta(self, layer: RasterLayer) -> dict:
        t = layer.item.transform()
        return {
            "name": layer.layer_name,
            "schema": layer.schema,
            "pos_x": layer.item.pos().x(),
            "pos_y": layer.item.pos().y(),
            "scale_x": t.m11(),
            "scale_y": t.m22(),
            "opacity": layer.item.opacity(),
            "visible": layer.item.isVisible(),
            "is_overlay": layer.is_overlay,
        }

    def _write_layer_to_zip(self, zf: zipfile.ZipFile, layer: RasterLayer, prefix: str) -> None:
        img = layer.item.pixmap().toImage()
        zf.writestr(f"{prefix}mask.tif", self._image_to_tif_bytes(img))
        zf.writestr(f"{prefix}layer.json", json.dumps(self._layer_to_meta(layer), indent=2))

    def _read_layer_from_zip(self, zf: zipfile.ZipFile, prefix: str) -> RasterLayer | None:
        try:
            tif_bytes = zf.read(f"{prefix}mask.tif")
            meta = json.loads(zf.read(f"{prefix}layer.json"))
        except KeyError:
            return None
        img = self._tif_bytes_to_image(tif_bytes)
        if img.isNull():
            return None
        item = QGraphicsPixmapItem(QPixmap.fromImage(img))
        item.setPos(meta.get("pos_x", 0.0), meta.get("pos_y", 0.0))
        sx, sy = meta.get("scale_x", 1.0), meta.get("scale_y", 1.0)
        if sx != 1.0 or sy != 1.0:
            item.setTransform(QTransform().scale(sx, sy))
        item.setOpacity(meta.get("opacity", 1.0))
        item.setVisible(meta.get("visible", True))
        assert self.scene is not None
        self.scene.addItem(item)
        return RasterLayer(
            layer_name=meta.get("name", "Layer"),
            schema=meta.get("schema", "unknown"),
            path=Path(meta.get("name", "Layer")),
            item=item,
            is_overlay=meta.get("is_overlay", False),
        )

    def _clear_all_layers(self) -> None:
        if self.scene is None or self.open_layers_list is None:
            return
        for layer in list(self.open_layers_list.raster_layers):
            self.scene.removeItem(layer.item)
        self.open_layers_list.clear_all()
        self._paint_image = None
        self._paint_item = None

    # -----------------------------
    # World / project metadata
    # -----------------------------
    def _new_project_flow(self) -> None:
        name = WorldDataDialog.get_world_data("", parent=self.window)
        if name is None:
            return
        self._clear_all_layers()
        self.undo_stack.clear()
        self._current_project_path = None
        self._update_world_name(name)
        self._status("New project created.", 2000)

    def _show_world_data_dialog(self) -> None:
        name = WorldDataDialog.get_world_data(self._world_name, parent=self.window)
        if name is None:
            return
        self._update_world_name(name)

    def _update_world_name(self, name: str) -> None:
        self._world_name = name
        display = f"MapCreator: {name}" if name else "MapCreator"
        if self.title_label is not None:
            self.title_label.setText(display)
        if self.window is not None:
            self.window.setWindowTitle(display)

    # -----------------------------
    # Save / open layer
    # -----------------------------
    def save_layer(self) -> None:
        if self.open_layers_list is None:
            return
        row = self.open_layers_list.currentRow()
        layers = self.open_layers_list.raster_layers
        if not (0 <= row < len(layers)):
            self._status("No layer selected to save.", 3000)
            return
        layer = layers[row]
        path_str, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save Layer",
            str(_dirs.DEFAULT_OPEN_DIR / f"{layer.layer_name}.mclayer"),
            "MapCreator Layer (*.mclayer);;All Files (*)",
        )
        if not path_str:
            return
        save_path = Path(path_str)
        if save_path.suffix.lower() != ".mclayer":
            save_path = save_path.with_suffix(".mclayer")
        try:
            with zipfile.ZipFile(save_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                self._write_layer_to_zip(zf, layer, "")
        except Exception as exc:
            error(f"Failed to save layer: {exc}")
            self._status(f"Error saving layer: {exc}", 5000)
            return
        self._status(f"Saved: {save_path.name}", 3000)

    def open_layer_file(self) -> None:
        if self.scene is None or self.open_layers_list is None:
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open Layer",
            str(_dirs.DEFAULT_OPEN_DIR),
            "MapCreator Layer (*.mclayer);;All Files (*)",
        )
        if not path_str:
            return
        try:
            with zipfile.ZipFile(Path(path_str), "r") as zf:
                layer = self._read_layer_from_zip(zf, "")
        except Exception as exc:
            error(f"Failed to open layer: {exc}")
            self._status(f"Error opening layer: {exc}", 5000)
            return
        if layer is None:
            self._status("Failed to load layer from file.", 5000)
            return
        self.open_layers_list.add_raster_layer(layer)

        if len(self.open_layers_list.raster_layers) == 1 and self.map_view is not None:
            self.map_view.fitInView(layer.item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._status(f"Opened layer: {layer.layer_name}", 3000)

    # -----------------------------
    # Save / open project
    # -----------------------------
    def save_project(self) -> None:
        """Save to the current project path; prompt via Save As if none is set."""
        if self._current_project_path is not None:
            self._do_save_project(self._current_project_path)
        else:
            self.save_project_as()

    def save_project_as(self) -> None:
        """Always prompt for a new path, then save."""
        if self.open_layers_list is None:
            return
        if not self.open_layers_list.raster_layers:
            self._status("No layers to save.", 3000)
            return
        default = (
            self._current_project_path
            if self._current_project_path is not None
            else _dirs.DEFAULT_OPEN_DIR / "project.mcproject"
        )
        path_str, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save Project As",
            str(default),
            "MapCreator Project (*.mcproject);;All Files (*)",
        )
        if not path_str:
            return
        save_path = Path(path_str)
        if save_path.suffix.lower() != ".mcproject":
            save_path = save_path.with_suffix(".mcproject")
        self._do_save_project(save_path)

    def _do_save_project(self, save_path: Path) -> None:
        layers = self.open_layers_list.raster_layers if self.open_layers_list else []
        if not layers:
            self._status("No layers to save.", 3000)
            return
        try:
            with zipfile.ZipFile(save_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("project.json", json.dumps(
                    {"world_name": self._world_name, "layer_count": len(layers)}, indent=2
                ))
                for i, layer in enumerate(layers):
                    self._write_layer_to_zip(zf, layer, f"layer_{i:04d}/")
        except Exception as exc:
            error(f"Failed to save project: {exc}")
            self._status(f"Error saving project: {exc}", 5000)
            return
        self._current_project_path = save_path
        self._status(f"Saved: {save_path.name} ({len(layers)} layers)", 3000)

    def open_project(self) -> None:
        if self.scene is None or self.open_layers_list is None:
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open Project",
            str(_dirs.DEFAULT_OPEN_DIR),
            "MapCreator Project (*.mcproject);;All Files (*)",
        )
        if not path_str:
            return
        try:
            with zipfile.ZipFile(Path(path_str), "r") as zf:
                meta = json.loads(zf.read("project.json"))
                layer_count = int(meta.get("layer_count", 0))
                self._clear_all_layers()
                for i in range(layer_count - 1, -1, -1):
                    layer = self._read_layer_from_zip(zf, f"layer_{i:04d}/")
                    if layer is not None:
                        self.open_layers_list.add_raster_layer(layer)
        except Exception as exc:
            error(f"Failed to open project: {exc}")
            self._status(f"Error opening project: {exc}", 5000)
            return
        self._current_project_path = Path(path_str)
        self._update_world_name(meta.get("world_name", ""))
        self.undo_stack.clear()

        layers = self.open_layers_list.raster_layers
        if layers and self.map_view is not None:
            bounds = self.scene.itemsBoundingRect()
            if not bounds.isNull():
                self.map_view.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
        self._status(f"Opened: {Path(path_str).name} ({len(layers)} layers)", 3000)

    # -----------------------------
    # Undo / redo
    # -----------------------------
    def _undo(self) -> None:
        desc = self.undo_stack.undo()
        self._status(f"Undo: {desc}" if desc else "Nothing to undo.", 1500)

    def _redo(self) -> None:
        desc = self.undo_stack.redo()
        self._status(f"Redo: {desc}" if desc else "Nothing to redo.", 1500)

    def _find_layer_row(self, layer: RasterLayer) -> int:
        if self.open_layers_list is None:
            return -1
        for i, l in enumerate(self.open_layers_list.raster_layers):
            if l is layer:
                return i
        return -1

    def _apply_paint_image(self, item: QGraphicsPixmapItem, img: QImage) -> None:
        """Restore img to item; also updates _paint_image if item is the active paint target."""
        item.setPixmap(QPixmap.fromImage(img))
        if item is self._paint_item:
            self._paint_image = img.copy()

    def _push_create_cmd(self, layer: RasterLayer, description: str) -> None:
        """Record an undo command for a layer that was just added to the scene and list."""
        saved_z = layer.item.zValue()

        def undo() -> None:
            row = self._find_layer_row(layer)
            if row < 0:
                return
            if layer.item is self._paint_item:
                self._paint_image = None
                self._paint_item = None
            if layer.is_overlay and self.map_view is not None:
                self.map_view.set_overlay_item(None)
            assert self.scene is not None
            assert self.open_layers_list is not None
            self.scene.removeItem(layer.item)
            self.open_layers_list.remove_layer(row)

        def redo() -> None:
            assert self.scene is not None
            assert self.open_layers_list is not None
            layer.item.setZValue(saved_z)
            self.scene.addItem(layer.item)
            self.open_layers_list.add_raster_layer(layer)

        self.undo_stack.push(UndoCommand(undo_fn=undo, redo_fn=redo, description=description))

    def _on_paint_ended(self) -> None:
        if self._stroke_start_image is None or self._paint_image is None or self._paint_item is None:
            self._stroke_start_image = None
            return
        before = self._stroke_start_image
        after  = self._paint_image.copy()
        item   = self._paint_item
        self._stroke_start_image = None
        self.undo_stack.push(UndoCommand(
            undo_fn=lambda: self._apply_paint_image(item, before),
            redo_fn=lambda: self._apply_paint_image(item, after),
            description="Paint stroke",
        ))

    def _on_layer_move_ended(self) -> None:
        if self._move_start_pos is None:
            return
        if self.open_layers_list is None:
            self._move_start_pos = None
            return
        row = self.open_layers_list.currentRow()
        layers = self.open_layers_list.raster_layers
        if not (0 <= row < len(layers)):
            self._move_start_pos = None
            return
        item = layers[row].item
        start_pos = self._move_start_pos
        end_pos   = item.pos()
        self._move_start_pos = None
        if start_pos == end_pos:
            return

        def apply_pos(pos: QPointF) -> None:
            item.setPos(pos)
            if item is self._paint_item:
                self._paint_origin = pos
            if self.map_view is not None:
                self.map_view.viewport().update()

        self.undo_stack.push(UndoCommand(
            undo_fn=lambda: apply_pos(start_pos),
            redo_fn=lambda: apply_pos(end_pos),
            description="Move layer",
        ))

    def _on_handle_drag_ended(self) -> None:
        if self._resize_start_state is None:
            return
        if self.open_layers_list is None:
            self._resize_start_state = None
            return
        row = self.open_layers_list.currentRow()
        layers = self.open_layers_list.raster_layers
        if not (0 <= row < len(layers)):
            self._resize_start_state = None
            return
        item = layers[row].item
        start_pos, start_tf = self._resize_start_state
        end_pos = item.pos()
        end_tf  = item.transform()
        self._resize_start_state = None

        def apply(pos: QPointF, tf: QTransform) -> None:
            item.setPos(pos)
            item.setTransform(tf)
            if self.map_view is not None:
                self.map_view.viewport().update()

        self.undo_stack.push(UndoCommand(
            undo_fn=lambda: apply(start_pos, start_tf),
            redo_fn=lambda: apply(end_pos, end_tf),
            description="Resize overlay",
        ))

    def _on_layer_renamed(self, layer: object, old_name: str, new_name: str) -> None:
        rl: RasterLayer = layer  # type: ignore[assignment]

        def apply_name(name: str) -> None:
            rl.layer_name = name
            if self.open_layers_list is None:
                return
            w = self.open_layers_list.get_row_widget(rl)
            if w is not None:
                w.set_name(name)

        self.undo_stack.push(UndoCommand(
            undo_fn=lambda: apply_name(old_name),
            redo_fn=lambda: apply_name(new_name),
            description=f"Rename to '{new_name}'",
        ))

    def _on_layer_reordered(self, old_order: list, new_order: list) -> None:
        self.undo_stack.push(UndoCommand(
            undo_fn=lambda: self.open_layers_list.restore_order(old_order),
            redo_fn=lambda: self.open_layers_list.restore_order(new_order),
            description="Reorder layers",
        ))

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
        """Open a raster file and add it as a layer using the currently selected schema."""
        assert self.window is not None

        # Refresh combobox so any YAML edits since startup are reflected
        self._populate_schema_combo()

        schema = (
            (self.palette_controls.current_schema if self.palette_controls else None)
            or "unknown"
        )
        self._status(f"Opening raster for schema '{schema}'…", 2000)
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
            str(_dirs.DEFAULT_OPEN_DIR),
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
            self._status(f"Failed to load image: {path.name}", 5000)
            return

        pix = QPixmap.fromImage(img)
        item = QGraphicsPixmapItem(pix)

        # Put new layers on top
        item.setZValue(len(self.scene.items()))

        self.scene.addItem(item)

        layer = RasterLayer(layer_name=name, path=path, item=item, schema=schema or "unknown")
        self.open_layers_list.add_raster_layer(layer)


        # Fit to view if first layer; otherwise preserve current zoom
        if len(self.open_layers_list.raster_layers) == 1:
            self.map_view.fitInView(item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self._push_create_cmd(layer, f"Open '{path.name}'")
        self._status(f"Loaded raster layer: {path.name}", 3000)

    # -----------------------------
    # Overlay
    # -----------------------------
    def open_overlay_file(self) -> None:
        """Open an image as a reference overlay (not paintable, movable)."""
        if self.scene is None or self.open_layers_list is None:
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open Overlay",
            str(_dirs.DEFAULT_OPEN_DIR),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        img = QImage(str(path))
        if img.isNull():
            self._status(f"Failed to load overlay: {path.name}", 5000)
            return
        item = QGraphicsPixmapItem(QPixmap.fromImage(img))
        item.setZValue(len(self.scene.items()))
        item.setOpacity(0.5)
        self.scene.addItem(item)
        layer = RasterLayer(
            layer_name=f"[Ref] {path.stem}",
            schema="overlay",
            path=path,
            item=item,
            is_overlay=True,
        )
        self.open_layers_list.add_raster_layer(layer)
        if len(self.open_layers_list.raster_layers) == 1 and self.map_view is not None:
            self.map_view.fitInView(item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._push_create_cmd(layer, f"Open overlay '{path.name}'")
        self._status(f"Opened overlay: {path.name}", 3000)

    # -----------------------------
    # Move mode
    # -----------------------------
    def _on_move_mode_changed(self, active: bool) -> None:
        if self.map_view is None:
            return
        if active:
            self.map_view.set_paint_mode(False)
            self.map_view.set_move_mode(True)
        else:
            self.map_view.set_move_mode(False)
            # Restore paint mode if a class or eraser was already active
            has_paint = (
                self._active_class_color is not None
                or (self.palette_controls is not None and self.palette_controls.is_eraser)
            )
            if has_paint:
                self.map_view.set_paint_mode(True)

    def _on_layer_moved(self, dx: float, dy: float) -> None:
        if self.open_layers_list is None:
            return
        row = self.open_layers_list.currentRow()
        layers = self.open_layers_list.raster_layers
        if not (0 <= row < len(layers)):
            return
        item = layers[row].item
        if self._move_start_pos is None:
            self._move_start_pos = item.pos()  # snapshot before first delta
        item.setPos(item.pos() + QPointF(dx, dy))
        if item is self._paint_item:
            self._paint_origin = item.pos()

    def _on_handle_dragged(self, handle: int, scene_pos: QPointF) -> None:
        """Resize the selected overlay by dragging one of its 8 handles."""
        if self.open_layers_list is None:
            return
        row = self.open_layers_list.currentRow()
        layers = self.open_layers_list.raster_layers
        if not (0 <= row < len(layers)) or not layers[row].is_overlay:
            return

        item = layers[row].item
        W0 = float(item.pixmap().width())
        H0 = float(item.pixmap().height())
        if W0 <= 0 or H0 <= 0:
            return

        if self._resize_start_state is None:
            self._resize_start_state = (item.pos(), item.transform())

        t   = item.transform()
        sx  = t.m11() if t.m11() != 0 else 1.0
        sy  = t.m22() if t.m22() != 0 else 1.0
        px, py       = item.pos().x(), item.pos().y()
        W,  H        = W0 * sx, H0 * sy
        right, bottom = px + W, py + H
        nx, ny       = scene_pos.x(), scene_pos.y()
        MIN = 10.0

        # For each handle: compute new (pos, sx, sy) keeping the opposite edge fixed.
        if handle == 0:    # NW — SE fixed
            nw = max(MIN, right  - nx);  nh = max(MIN, bottom - ny)
            item.setPos(right - nw, bottom - nh)
            item.setTransform(QTransform().scale(nw / W0, nh / H0))
        elif handle == 1:  # N  — bottom fixed
            nh = max(MIN, bottom - ny)
            item.setPos(px, bottom - nh)
            item.setTransform(QTransform().scale(sx, nh / H0))
        elif handle == 2:  # NE — SW fixed
            nw = max(MIN, nx - px);  nh = max(MIN, bottom - ny)
            item.setPos(px, bottom - nh)
            item.setTransform(QTransform().scale(nw / W0, nh / H0))
        elif handle == 3:  # W  — right fixed
            nw = max(MIN, right - nx)
            item.setPos(right - nw, py)
            item.setTransform(QTransform().scale(nw / W0, sy))
        elif handle == 4:  # E  — left fixed
            nw = max(MIN, nx - px)
            item.setTransform(QTransform().scale(nw / W0, sy))
        elif handle == 5:  # SW — NE fixed
            nw = max(MIN, right  - nx);  nh = max(MIN, ny - py)
            item.setPos(right - nw, py)
            item.setTransform(QTransform().scale(nw / W0, nh / H0))
        elif handle == 6:  # S  — top fixed
            nh = max(MIN, ny - py)
            item.setTransform(QTransform().scale(sx, nh / H0))
        elif handle == 7:  # SE — NW fixed
            nw = max(MIN, nx - px);  nh = max(MIN, ny - py)
            item.setTransform(QTransform().scale(nw / W0, nh / H0))

    # -----------------------------
    # Canvas toolbar handlers
    # -----------------------------
    def _on_snap_mode_changed(self, active: bool) -> None:
        if self.map_view is not None:
            self.map_view.set_snap_mode(active)

    def _on_grid_toggled(self, checked: bool) -> None:
        assert self._top_options_bar is not None
        assert self._grid_size_combo is not None
        self._top_options_bar.setVisible(checked)
        size = int(self._grid_size_combo.currentText())
        self.map_view.set_grid(checked, size)
        if self.palette_controls is not None:
            self.palette_controls.set_grid_active(checked)

    def _on_grid_size_changed(self, text: str) -> None:
        assert self._grid_toggle_btn is not None
        if not self._grid_toggle_btn.isChecked():
            return
        try:
            self.map_view.set_grid(True, int(text))
        except ValueError:
            pass

    def _on_zoom_changed(self, factor: float) -> None:
        assert self._zoom_label is not None
        self._zoom_label.setText(f"{int(round(factor * 100))}%")

    def _on_cursor_moved(self, pos: QPointF) -> None:
        assert self._coord_label is not None
        self._coord_label.setText(f"x: {int(pos.x())}   y: {int(pos.y())}")

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

        print(f"[STATUS] {msg}")


if __name__ == "__main__":
    MapCreator().run()
