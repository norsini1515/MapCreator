"""
mapcreator.features.overlays.map_creator

Defines MapCreator: the main application/controller for the MapCreator UI.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, cast

from mapcreator.features.overlays.undo_stack import UndoStack, UndoCommand

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QBuffer, QByteArray, QFile, Qt, QIODevice, QPointF, QRect, QSize
from PySide6.QtGui import (
    QAction, QBrush, QColor, QImage, QKeySequence, QPainter,
    QPixmap, QRegion, QShortcut, QTransform,
)
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDockWidget,
    QFileDialog, QFormLayout, QFrame, QGraphicsPixmapItem, QGraphicsScene,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QSizePolicy,
    QStackedWidget, QStyle, QTabWidget, QToolBar, QToolButton, QVBoxLayout,
    QWidget,
)

from mapcreator import directories as _dirs
from mapcreator.features.overlays.layer_view import LayerView
from mapcreator.features.overlays.open_layers_list import OpenLayersList, RasterLayer
from mapcreator.features.overlays.palette_controls import PaletteControls
from mapcreator.globals.config_models import read_config_file
from mapcreator.globals.logutil import error


# ---------------------------------------------------------------------------
# Per-tab state
# ---------------------------------------------------------------------------

@dataclass
class EditTab:
    """All state that belongs to one named edit context."""
    name: str
    scene: QGraphicsScene
    view: LayerView
    layers_list: OpenLayersList
    undo_stack: UndoStack
    # Paint state
    paint_image: Optional[QImage] = None
    paint_item: Optional[QGraphicsPixmapItem] = None
    paint_origin: QPointF = field(default_factory=QPointF)
    stroke_start_image: Optional[QImage] = None
    move_start_pos: Optional[QPointF] = None
    resize_start_state: Optional[tuple] = None
    # Active brush
    active_schema: Optional[str] = None
    active_class_id: Optional[int] = None
    active_class_color: Optional[str] = None
    active_class_name: Optional[str] = None
    paint_layer_counts: dict = field(default_factory=dict)
    # Canvas toolbar (set by _build_tab_canvas)
    grid_toggle_btn: Optional[QToolButton] = None
    grid_size_combo: Optional[QComboBox] = None
    zoom_label: Optional[QLabel] = None
    coord_label: Optional[QLabel] = None
    top_options_bar: Optional[QFrame] = None


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------

class WorldDataDialog(QDialog):
    """Popup for editing project-level world metadata."""

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
        dlg = WorldDataDialog(world_name, parent=parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.world_name
        return None


# ---------------------------------------------------------------------------
# Custom QUiLoader
# ---------------------------------------------------------------------------

class Loader(QUiLoader):
    def createWidget(self, className, parent=None, name=""):
        if className == "LayerView":
            w = LayerView(parent)
            w.setObjectName(name)
            return w
        if className == "OpenLayersList":
            w = OpenLayersList(parent)
            w.setObjectName(name)
            return w
        return super().createWidget(className, parent, name)


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class MapCreator:
    """Main application controller."""

    _BRUSH_CLIP_TOL: int = 12

    def __init__(self, class_cfg=None) -> None:
        self.qt_app: QApplication = QApplication([])
        self.window: Optional[QMainWindow] = None

        # Docks
        self.layer_control_dock: Optional[QDockWidget] = None
        self.open_layers_dock: Optional[QDockWidget] = None
        self.palette_controls: Optional[PaletteControls] = None

        # Tab management
        self._tabs: list[EditTab] = []
        self._active_tab_idx: int = 0
        self._tab_widget: Optional[QTabWidget] = None
        self._layers_stack: Optional[QStackedWidget] = None

        # Project state
        self._world_name: str = ""
        self._current_project_path: Optional[Path] = None
        self.title_label: Optional[QLabel] = None

        # Actions
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
        self.action_new_edit_tab: Optional[QAction] = None

        self.class_cfg = class_cfg or read_config_file(None, kind="class_configs")

        self._build_ui()
        self._wire_events()

    # ------------------------------------------------------------------
    # Active tab shortcut
    # ------------------------------------------------------------------

    @property
    def _tab(self) -> EditTab:
        return self._tabs[self._active_tab_idx]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
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

        # Grab the initial LayerView and OpenLayersList from the .ui
        initial_view = self.window.findChild(LayerView, "LayerView")
        if initial_view is None:
            raise RuntimeError("Could not find LayerView widget in .ui")

        initial_layers_list = self.window.findChild(OpenLayersList, "OpenLayersList")
        if initial_layers_list is None:
            raise RuntimeError("Could not find OpenLayersList widget in .ui")

        # Actions
        self.action_new_raster = self.window.findChild(QAction, "actionNew_Layer")
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
        self.action_new_edit_tab = self.window.findChild(QAction, "actionNew_Edit_Tab")

        # Replace the single LayerView in central layout with a QTabWidget
        central = self.window.centralWidget()
        central_layout = central.layout()
        central_layout.removeWidget(initial_view)

        self._tab_widget = QTabWidget(central)
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.setMovable(False)
        self._tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        add_tab_btn = QToolButton()
        add_tab_btn.setText("+")
        add_tab_btn.setToolTip("New Edit Tab")
        add_tab_btn.clicked.connect(self._prompt_new_tab)
        self._tab_widget.setCornerWidget(add_tab_btn, Qt.Corner.TopRightCorner)

        central_layout.addWidget(self._tab_widget, 1)

        # Replace the OpenLayersList in the dock with a QStackedWidget
        dock_contents = self.window.findChild(QWidget, "dockWidgetContents_Layers")
        outer_layout = dock_contents.layout()
        inner_layout = outer_layout.itemAt(0).layout() if outer_layout else None

        self._layers_stack = QStackedWidget()
        # Reparenting initial_layers_list into _layers_stack removes it from the old layout
        self._layers_stack.addWidget(initial_layers_list)
        if inner_layout is not None:
            inner_layout.addWidget(self._layers_stack, 1)
        else:
            outer_layout.addWidget(self._layers_stack, 1)

        # Docks
        self.layer_control_dock = self.window.findChild(QDockWidget, "layer_palette_controls_dock")
        self.open_layers_dock = self.window.findChild(QDockWidget, "OpenLayersListDock")

        if self.layer_control_dock is not None and self.open_layers_dock is not None:
            self.window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.layer_control_dock)
            self.window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.open_layers_dock)
            self.window.splitDockWidget(self.layer_control_dock, self.open_layers_dock, Qt.Orientation.Vertical)

        self.palette_controls = PaletteControls(main_window=self.window)

        # Layers-panel toolbar
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
        if dock_contents is not None and dock_contents.layout() is not None:
            dock_contents.layout().addWidget(layers_toolbar)

        # Build the first tab using the .ui widgets
        self._create_tab_from_widgets("Terrain", initial_view, initial_layers_list)

    def _build_tab_canvas(self, tab: EditTab) -> QWidget:
        """Wrap tab.view in a canvas frame with top options bar and bottom toolbar."""
        assert self.window is not None

        canvas_frame = QFrame(self._tab_widget)
        canvas_frame.setObjectName("canvas_frame")
        canvas_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas_frame.setStyleSheet(
            "QFrame#canvas_frame { background: #ebebeb; border: 1px solid #c8c8c8; border-radius: 3px; }"
        )
        canvas_vbox = QVBoxLayout(canvas_frame)
        canvas_vbox.setContentsMargins(8, 8, 8, 0)
        canvas_vbox.setSpacing(0)

        # Top options strip (grid size)
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
        grid_size_combo = QComboBox()
        grid_size_combo.addItems(["50", "100", "200"])
        grid_size_combo.setCurrentIndex(1)
        grid_size_combo.setFixedWidth(64)
        top_layout.addWidget(grid_size_label)
        top_layout.addWidget(grid_size_combo)
        top_layout.addStretch()
        top_bar.hide()
        canvas_vbox.addWidget(top_bar)

        canvas_vbox.addWidget(tab.view, 1)

        # Bottom toolbar
        bottom_bar = QFrame(canvas_frame)
        bottom_bar.setObjectName("canvas_bottom_bar")
        bottom_bar.setFrameShape(QFrame.Shape.NoFrame)
        bottom_bar.setStyleSheet(
            "QFrame#canvas_bottom_bar { background: #f0f0f0; border-top: 1px solid #cccccc; }"
        )
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(6, 2, 6, 2)
        bottom_layout.setSpacing(8)

        grid_toggle_btn = QToolButton()
        grid_toggle_btn.setText("Grid")
        grid_toggle_btn.setCheckable(True)
        grid_toggle_btn.setToolTip("Toggle grid overlay")
        bottom_layout.addWidget(grid_toggle_btn)

        def _vsep() -> QFrame:
            s = QFrame()
            s.setFrameShape(QFrame.Shape.VLine)
            s.setFrameShadow(QFrame.Shadow.Sunken)
            return s

        bottom_layout.addWidget(_vsep())
        zoom_label = QLabel("100%")
        zoom_label.setToolTip("Zoom level")
        zoom_label.setMinimumWidth(44)
        bottom_layout.addWidget(zoom_label)
        bottom_layout.addWidget(_vsep())
        coord_label = QLabel("x: —   y: —")
        coord_label.setToolTip("Scene coordinates under cursor")
        bottom_layout.addWidget(coord_label)
        bottom_layout.addStretch()
        canvas_vbox.addWidget(bottom_bar)

        # Store toolbar refs in the tab
        tab.grid_toggle_btn = grid_toggle_btn
        tab.grid_size_combo = grid_size_combo
        tab.zoom_label = zoom_label
        tab.coord_label = coord_label
        tab.top_options_bar = top_bar

        return canvas_frame

    def _create_tab_from_widgets(self, name: str, view: LayerView, layers_list: OpenLayersList) -> EditTab:
        """Create the first tab using widgets that came from the .ui file."""
        assert self._tab_widget is not None
        assert self._layers_stack is not None

        scene = QGraphicsScene(self.window)
        scene.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        view.setScene(scene)

        tab = EditTab(name=name, scene=scene, view=view, layers_list=layers_list, undo_stack=UndoStack())
        canvas_frame = self._build_tab_canvas(tab)
        self._tab_widget.addTab(canvas_frame, name)
        # layers_list was already added to _layers_stack in _build_ui

        self._tabs.append(tab)
        return tab

    def _add_tab(self, name: str) -> EditTab:
        """Create a brand-new edit tab."""
        assert self._tab_widget is not None
        assert self._layers_stack is not None
        assert self.window is not None

        scene = QGraphicsScene(self.window)
        scene.setBackgroundBrush(QBrush(QColor(255, 255, 255)))

        view = LayerView(self._tab_widget)
        view.setObjectName(f"LayerView_tab_{len(self._tabs)}")
        view.setScene(scene)
        if self.palette_controls is not None:
            view.set_brush(self.palette_controls.brush_size, self.palette_controls.brush_shape)

        layers_list = OpenLayersList()
        layers_list.setObjectName(f"OpenLayersList_tab_{len(self._tabs)}")

        tab = EditTab(name=name, scene=scene, view=view, layers_list=layers_list, undo_stack=UndoStack())
        canvas_frame = self._build_tab_canvas(tab)

        self._tab_widget.addTab(canvas_frame, name)
        self._layers_stack.addWidget(layers_list)
        self._tabs.append(tab)

        self._wire_tab_events(tab)

        idx = len(self._tabs) - 1
        self._tab_widget.setCurrentIndex(idx)
        return tab

    def _prompt_new_tab(self) -> None:
        assert self.window is not None
        name, ok = QInputDialog.getText(self.window, "New Edit Tab", "Tab name:", text="New Tab")
        if ok and name.strip():
            self._add_tab(name.strip())

    def _on_tab_changed(self, idx: int) -> None:
        if not self._tabs or not (0 <= idx < len(self._tabs)):
            return
        self._active_tab_idx = idx
        tab = self._tabs[idx]

        if self._layers_stack is not None:
            self._layers_stack.setCurrentWidget(tab.layers_list)

        # Refresh palette to this tab's schema
        if tab.active_schema and self.palette_controls is not None:
            defines = self.class_cfg.registry.defines.get(tab.active_schema, {})
            raw = {cid: {"name": cd.name, "color": cd.color} for cid, cd in defines.items()}
            self.palette_controls.set_palette_defines(raw)
            self.palette_controls.set_schema_label(tab.active_schema)
            self.palette_controls.set_schema(tab.active_schema)
        elif self.palette_controls is not None and not tab.active_schema:
            self.palette_controls.clear_brush()

    def _on_tab_close_requested(self, idx: int) -> None:
        if len(self._tabs) <= 1:
            self._status("Cannot close the last tab.", 2000)
            return
        self._close_tab(idx)

    def _close_tab(self, idx: int) -> None:
        assert self._tab_widget is not None
        assert self._layers_stack is not None

        tab = self._tabs[idx]
        for layer in list(tab.layers_list.raster_layers):
            tab.scene.removeItem(layer.item)
        tab.layers_list.clear_all()

        self._tab_widget.blockSignals(True)
        self._tab_widget.removeTab(idx)
        self._layers_stack.removeWidget(tab.layers_list)
        self._tabs.pop(idx)
        self._tab_widget.blockSignals(False)

        new_idx = min(idx, len(self._tabs) - 1)
        self._tab_widget.setCurrentIndex(new_idx)
        self._active_tab_idx = new_idx
        self._on_tab_changed(new_idx)

        tab.layers_list.deleteLater()

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    def _wire_events(self) -> None:
        assert self.window is not None
        assert self._tab_widget is not None

        # Tab switching
        self._tab_widget.currentChanged.connect(self._on_tab_changed)
        self._tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)

        # Palette controls → active tab (evaluated at call time via self._tab)
        assert self.palette_controls is not None
        self.palette_controls.classSelected.connect(self._on_class_selected)
        self.palette_controls.brushCleared.connect(self._on_brush_cleared)
        self.palette_controls.brushSettingsChanged.connect(
            lambda sz, sh: self._tab.view.set_brush(sz, sh)
        )
        self.palette_controls.schemaComboChanged.connect(self._on_layer_schema_changed)
        self.palette_controls.snapToGridChanged.connect(
            lambda active: self._tab.view.set_snap_mode(active)
        )
        self.palette_controls.moveModeChanged.connect(self._on_move_mode_changed)

        # Wire first tab (already added in _build_ui)
        self._wire_tab_events(self._tabs[0])
        if self.palette_controls is not None:
            self._tabs[0].view.set_brush(
                self.palette_controls.brush_size, self.palette_controls.brush_shape
            )

        # Global action wiring
        if self.action_new_raster is not None:
            self.action_new_raster.triggered.connect(self.new_layer_flow)
        if self.action_new_layer_mask is not None:
            self.action_new_layer_mask.triggered.connect(self._new_layer_mask)
        if self.action_open_overlay is not None:
            self.action_open_overlay.triggered.connect(self.open_overlay_file)
        if self.action_new_edit_tab is not None:
            self.action_new_edit_tab.triggered.connect(self._prompt_new_tab)
        menu_open_overlay = self.window.findChild(QAction, "actionOpen_Overlay")
        if menu_open_overlay is not None:
            menu_open_overlay.triggered.connect(self.open_overlay_file)
        if self.action_new_project is not None:
            self.action_new_project.triggered.connect(self._new_project_flow)
        if self.action_world_data is not None:
            self.action_world_data.triggered.connect(self._show_world_data_dialog)
        if self.action_save_layer is not None:
            self.action_save_layer.triggered.connect(self.save_layer)
        if self.action_open_layer is not None:
            self.action_open_layer.triggered.connect(self.open_layer_file)
        if self.action_save_project is not None:
            self.action_save_project.triggered.connect(self.save_project)
        if self.action_save_project_as is not None:
            self.action_save_project_as.triggered.connect(self.save_project_as)
        if self.action_open_project is not None:
            self.action_open_project.triggered.connect(self.open_project)

        # Undo / redo
        QShortcut(QKeySequence.StandardKey.Undo, self.window).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self.window).activated.connect(self._redo)

        # Dock visibility toggles
        if self.action_layer_list is not None and self.open_layers_dock is not None:
            self.action_layer_list.setCheckable(True)
            self.action_layer_list.setChecked(self.open_layers_dock.isVisible())
            self.action_layer_list.toggled.connect(self.open_layers_dock.setVisible)
            self.open_layers_dock.visibilityChanged.connect(self.action_layer_list.setChecked)
        if self.action_palette_controls is not None and self.layer_control_dock is not None:
            self.action_palette_controls.setCheckable(True)
            self.action_palette_controls.setChecked(self.layer_control_dock.isVisible())
            self.action_palette_controls.toggled.connect(self.layer_control_dock.setVisible)
            self.layer_control_dock.visibilityChanged.connect(self.action_palette_controls.setChecked)

        self._populate_schema_combo()

    def _wire_tab_events(self, tab: EditTab) -> None:
        """Wire a tab's view and layer list signals."""
        tab.layers_list.layerSchemaChanged.connect(self._on_layer_schema_changed)
        tab.layers_list.layerSelected.connect(self._sync_paint_target)
        tab.layers_list.duplicateRequested.connect(self._on_duplicate_layer)
        tab.layers_list.deleteRequested.connect(self._on_delete_layer)
        tab.layers_list.mergeRequested.connect(self._on_merge_layers)
        tab.layers_list.layerRenamed.connect(self._on_layer_renamed)
        tab.layers_list.reorderOccurred.connect(self._on_layer_reordered)

        tab.view.paintStroke.connect(self._on_paint_stroke)
        tab.view.paintEnded.connect(self._on_paint_ended)
        tab.view.layerMoved.connect(self._on_layer_moved)
        tab.view.layerMoveEnded.connect(self._on_layer_move_ended)
        tab.view.handleDragged.connect(self._on_handle_dragged)
        tab.view.handleDragEnded.connect(self._on_handle_drag_ended)

        # Canvas toolbar
        assert tab.grid_toggle_btn is not None
        assert tab.grid_size_combo is not None
        assert tab.zoom_label is not None
        assert tab.coord_label is not None

        tab.grid_toggle_btn.toggled.connect(lambda checked, t=tab: self._on_grid_toggled(t, checked))
        tab.grid_size_combo.currentTextChanged.connect(lambda text, t=tab: self._on_grid_size_changed(t, text))
        tab.view.zoomChanged.connect(lambda f, t=tab: t.zoom_label.setText(f"{int(round(f * 100))}%"))
        tab.view.cursorMoved.connect(
            lambda pos, t=tab: t.coord_label.setText(f"x: {int(pos.x())}   y: {int(pos.y())}")
        )
        tab.view.cursorLeft.connect(lambda t=tab: t.coord_label.setText("x: —   y: —"))

        tab.view.zoomChanged.emit(tab.view.transform().m11())

    # ------------------------------------------------------------------
    # Schema / palette
    # ------------------------------------------------------------------

    def _populate_schema_combo(self) -> None:
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
        tab = self._tab
        if schema == tab.active_schema:
            return
        tab.active_schema = schema
        self._on_brush_cleared()
        if self.palette_controls is None:
            return
        defines = self.class_cfg.registry.defines.get(schema, {})
        raw = {cid: {"name": cd.name, "color": cd.color} for cid, cd in defines.items()}
        self.palette_controls.set_palette_defines(raw)
        self.palette_controls.set_schema_label(schema)
        self.palette_controls.set_schema(schema)

    def _sync_paint_target(self, row: int) -> None:
        tab = self._tab
        layers = tab.layers_list.raster_layers
        if row < 0 or row >= len(layers):
            tab.paint_image = None
            tab.paint_item = None
            return
        layer = layers[row]
        item = layer.item
        tab.paint_item = item
        tab.paint_origin = item.pos()
        if layer.is_overlay:
            tab.paint_image = None
            tab.view.set_overlay_item(item)
        else:
            tab.paint_image = item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
            tab.view.set_overlay_item(None)

    def _on_class_selected(self, class_id: int) -> None:
        tab = self._tab
        if tab.active_schema is None:
            return
        try:
            class_def = self.class_cfg.registry.get(tab.active_schema, class_id)
        except KeyError:
            return
        tab.active_class_id = class_id
        tab.active_class_color = class_def.color
        tab.active_class_name = class_def.name
        tab.view.set_paint_mode(True)

    def _on_brush_cleared(self) -> None:
        tab = self._tab
        tab.active_class_id = None
        tab.active_class_color = None
        tab.active_class_name = None
        is_eraser = self.palette_controls is not None and self.palette_controls.is_eraser
        if not is_eraser:
            tab.view.set_paint_mode(False)

    # ------------------------------------------------------------------
    # Apply clip
    # ------------------------------------------------------------------

    def _build_apply_clip(self, tab: EditTab, cx: int, cy: int, r: int, shape: str) -> QRegion | None:
        if self.palette_controls is None:
            return None
        disabled = self.palette_controls.get_disabled_classes()
        if not disabled:
            return None

        ref_layers = [
            layer for layer in tab.layers_list.raster_layers
            if layer.item.isVisible() and layer.item is not tab.paint_item
        ]
        if not ref_layers:
            return None

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

        cache: list[tuple] = []
        for ref_layer in ref_layers:
            img = ref_layer.item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
            ref_origin = ref_layer.item.pos()
            off_x = int(tab.paint_origin.x() - ref_origin.x())
            off_y = int(tab.paint_origin.y() - ref_origin.y())
            cache.append((img, off_x, off_y, img.width(), img.height()))

        tol = self._BRUSH_CLIP_TOL
        region = QRegion()

        for dy in range(-r, r + 1):
            py = cy + dy
            x_half = int(math.sqrt(max(0.0, r * r - dy * dy))) if shape != "Square" else r
            run_start: int | None = None
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
                            continue
                        s = QColor(img.pixel(rx, ry))
                        if s.alpha() < 64:
                            continue
                        sr, sg, sb = s.red(), s.green(), s.blue()
                        if any(
                            abs(sr - pr) < tol and abs(sg - pg) < tol and abs(sb - pb) < tol
                            for pr, pg, pb in protected
                        ):
                            allowed = False
                            break
                if allowed and run_start is None:
                    run_start = px
                elif not allowed and run_start is not None:
                    region = region.united(QRegion(QRect(run_start, py, px - run_start, 1)))
                    run_start = None

        return region

    # ------------------------------------------------------------------
    # Paint stroke handlers
    # ------------------------------------------------------------------

    def _on_paint_stroke(self, pos: QPointF) -> None:
        tab = self._tab
        if tab.paint_image is None:
            return
        is_eraser = self.palette_controls is not None and self.palette_controls.is_eraser
        if not is_eraser and tab.active_class_color is None:
            return

        if tab.stroke_start_image is None:
            tab.stroke_start_image = tab.paint_image.copy()

        if self.palette_controls is not None and self.palette_controls.snap_to_grid:
            self._paint_snap_cell(tab, pos, is_eraser)
            return

        r = self.palette_controls.brush_size if self.palette_controls else 8
        shape = self.palette_controls.brush_shape if self.palette_controls else "Circle"

        img_x = int(pos.x() - tab.paint_origin.x())
        img_y = int(pos.y() - tab.paint_origin.y())

        clip = self._build_apply_clip(tab, img_x, img_y, r, shape)

        painter = QPainter(tab.paint_image)
        painter.setPen(Qt.PenStyle.NoPen)
        if clip is not None:
            if clip.isEmpty():
                return
            painter.setClipRegion(clip)

        if is_eraser:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.setBrush(QBrush(QColor(0, 0, 0, 255)))
        else:
            painter.setBrush(QBrush(QColor(tab.active_class_color)))

        if shape == "Square":
            painter.drawRect(img_x - r, img_y - r, r * 2, r * 2)
        else:
            painter.drawEllipse(img_x - r, img_y - r, r * 2, r * 2)
        painter.end()

        assert tab.paint_item is not None
        tab.paint_item.setPixmap(QPixmap.fromImage(tab.paint_image))

    def _paint_snap_cell(self, tab: EditTab, pos: QPointF, is_eraser: bool) -> None:
        if tab.paint_image is None or tab.paint_item is None:
            return
        bounds = tab.scene.itemsBoundingRect()
        if bounds.isNull():
            return

        count = int(tab.grid_size_combo.currentText()) if tab.grid_size_combo else 100
        cell_w = bounds.width() / count
        cell_h = bounds.height() / count
        if cell_w < 1 or cell_h < 1:
            return

        col = max(0, min(count - 1, int((pos.x() - bounds.left()) / cell_w)))
        row = max(0, min(count - 1, int((pos.y() - bounds.top()) / cell_h)))

        img_x = int(bounds.left() + col * cell_w - tab.paint_origin.x())
        img_y = int(bounds.top() + row * cell_h - tab.paint_origin.y())
        img_w = max(1, round(cell_w))
        img_h = max(1, round(cell_h))

        cell_cx = img_x + img_w // 2
        cell_cy = img_y + img_h // 2
        r = max(img_w, img_h) // 2 + 1
        clip = self._build_apply_clip(tab, cell_cx, cell_cy, r, "Square")

        painter = QPainter(tab.paint_image)
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
            painter.setBrush(QBrush(QColor(tab.active_class_color)))

        painter.drawRect(img_x, img_y, img_w, img_h)
        painter.end()
        tab.paint_item.setPixmap(QPixmap.fromImage(tab.paint_image))

    # ------------------------------------------------------------------
    # Layer operations (duplicate / delete / merge / new mask)
    # ------------------------------------------------------------------

    def _on_duplicate_layer(self, row: int) -> None:
        tab = self._tab
        layers = tab.layers_list.raster_layers
        if not (0 <= row < len(layers)):
            return

        src = layers[row]
        new_item = QGraphicsPixmapItem(src.item.pixmap().copy())
        new_item.setPos(src.item.pos())
        new_item.setVisible(src.item.isVisible())
        new_item.setOpacity(src.item.opacity())
        new_item.setZValue(src.item.zValue() + 1)
        tab.scene.addItem(new_item)

        layer = RasterLayer(
            layer_name=f"Copy of {src.layer_name}",
            schema=src.schema,
            path=src.path,
            item=new_item,
        )
        tab.layers_list.add_raster_layer(layer)
        self._push_create_cmd(tab, layer, f"Duplicate '{layer.layer_name}'")
        self._status(f"Duplicated: {layer.layer_name}", 2000)

    def _on_delete_layer(self, row: int) -> None:
        tab = self._tab
        layers = tab.layers_list.raster_layers
        if not (0 <= row < len(layers)):
            return
        layer = layers[row]
        saved_z = layer.item.zValue()

        if tab.paint_item is layer.item:
            tab.paint_image = None
            tab.paint_item = None
        if layer.is_overlay:
            tab.view.set_overlay_item(None)
        tab.scene.removeItem(layer.item)
        tab.layers_list.remove_layer(row)

        def undo_delete() -> None:
            layer.item.setZValue(saved_z)
            tab.scene.addItem(layer.item)
            tab.layers_list.add_raster_layer(layer)

        def redo_delete() -> None:
            r = self._find_layer_row_in(tab, layer)
            if r < 0:
                return
            if layer.item is tab.paint_item:
                tab.paint_image = None
                tab.paint_item = None
            if layer.is_overlay:
                tab.view.set_overlay_item(None)
            tab.scene.removeItem(layer.item)
            tab.layers_list.remove_layer(r)

        tab.undo_stack.push(UndoCommand(
            undo_fn=undo_delete, redo_fn=redo_delete,
            description=f"Delete '{layer.layer_name}'"
        ))
        self._status(f"Deleted layer: {layer.layer_name}", 2000)

    def _on_merge_layers(self, rows: list) -> None:
        tab = self._tab
        layers = tab.layers_list.raster_layers
        valid = [r for r in rows if 0 <= r < len(layers)]
        if len(valid) < 2:
            return

        selected = [layers[r] for r in valid]
        if any(l.is_overlay for l in selected):
            self._status("Cannot merge overlay layers.", 3000)
            return

        pre_merge_order = list(tab.layers_list.raster_layers)
        selected_sorted = sorted(selected, key=lambda l: l.item.zValue())
        top_layer = selected_sorted[-1]

        bounds = tab.scene.itemsBoundingRect()
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
        tab.scene.addItem(new_item)

        new_layer = RasterLayer(
            layer_name=f"Merged: {top_layer.layer_name}",
            schema=top_layer.schema,
            path=Path("merged"),
            item=new_item,
        )

        for row in sorted(valid, reverse=True):
            layer = layers[row]
            if layer.item is tab.paint_item:
                tab.paint_image = None
                tab.paint_item = None
            tab.scene.removeItem(layer.item)
            tab.layers_list.remove_layer(row)

        tab.layers_list.add_raster_layer(new_layer)
        post_merge_order = list(tab.layers_list.raster_layers)

        removed = [l for l in pre_merge_order if l not in post_merge_order]
        added   = [l for l in post_merge_order if l not in pre_merge_order]

        def undo_merge() -> None:
            for l in added:
                r = self._find_layer_row_in(tab, l)
                if r >= 0:
                    if l.item is tab.paint_item:
                        tab.paint_image = None
                        tab.paint_item = None
                    tab.scene.removeItem(l.item)
            for l in removed:
                tab.scene.addItem(l.item)
            tab.layers_list.restore_order(pre_merge_order)

        def redo_merge() -> None:
            for l in removed:
                r = self._find_layer_row_in(tab, l)
                if r >= 0:
                    if l.item is tab.paint_item:
                        tab.paint_image = None
                        tab.paint_item = None
                    tab.scene.removeItem(l.item)
            for l in added:
                tab.scene.addItem(l.item)
            tab.layers_list.restore_order(post_merge_order)

        tab.undo_stack.push(UndoCommand(
            undo_fn=undo_merge, redo_fn=redo_merge,
            description=f"Merge {len(valid)} layers"
        ))
        self._status(f"Merged {len(valid)} layers → {new_layer.layer_name}", 3000)

    def _new_layer_mask(self) -> None:
        tab = self._tab
        schema = (
            (self.palette_controls.current_schema if self.palette_controls else None)
            or tab.active_schema
            or "unknown"
        )
        name_base = tab.active_class_name or schema or "Paint"
        count = tab.paint_layer_counts.get(name_base, 0) + 1
        tab.paint_layer_counts[name_base] = count

        existing = tab.scene.itemsBoundingRect()
        if existing.isNull():
            vp = tab.view.viewport()
            w, h = vp.width(), vp.height()
            origin = tab.view.mapToScene(0, 0)
        else:
            w, h = int(existing.width()), int(existing.height())
            origin = existing.topLeft()

        blank = QImage(w, h, QImage.Format.Format_ARGB32)
        blank.fill(Qt.GlobalColor.transparent)
        item = QGraphicsPixmapItem(QPixmap.fromImage(blank))
        item.setPos(origin)
        item.setZValue(999)
        tab.scene.addItem(item)

        layer_name = f"New {name_base} Layer {count}"
        layer = RasterLayer(layer_name=layer_name, schema=schema, path=Path(layer_name), item=item)
        tab.layers_list.add_raster_layer(layer)
        self._push_create_cmd(tab, layer, f"New layer '{layer_name}'")
        self._status(f"Created mask layer: {layer_name}")

    # ------------------------------------------------------------------
    # Save / load helpers
    # ------------------------------------------------------------------

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

    def _read_layer_from_zip(self, zf: zipfile.ZipFile, prefix: str, scene: QGraphicsScene) -> RasterLayer | None:
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
        scene.addItem(item)
        return RasterLayer(
            layer_name=meta.get("name", "Layer"),
            schema=meta.get("schema", "unknown"),
            path=Path(meta.get("name", "Layer")),
            item=item,
            is_overlay=meta.get("is_overlay", False),
        )

    def _clear_tab_layers(self, tab: EditTab) -> None:
        for layer in list(tab.layers_list.raster_layers):
            tab.scene.removeItem(layer.item)
        tab.layers_list.clear_all()
        tab.paint_image = None
        tab.paint_item = None

    # ------------------------------------------------------------------
    # Project / world metadata
    # ------------------------------------------------------------------

    def _new_project_flow(self) -> None:
        name = WorldDataDialog.get_world_data("", parent=self.window)
        if name is None:
            return
        # Reset to a single tab
        assert self._tab_widget is not None
        while len(self._tabs) > 1:
            self._close_tab(len(self._tabs) - 1)
        self._tab_widget.setTabText(0, "Terrain")
        self._tabs[0].name = "Terrain"
        self._clear_tab_layers(self._tabs[0])
        self._tabs[0].undo_stack.clear()
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

    # ------------------------------------------------------------------
    # Save / open layer
    # ------------------------------------------------------------------

    def save_layer(self) -> None:
        tab = self._tab
        row = tab.layers_list.currentRow()
        layers = tab.layers_list.raster_layers
        if not (0 <= row < len(layers)):
            self._status("No layer selected to save.", 3000)
            return
        layer = layers[row]
        path_str, _ = QFileDialog.getSaveFileName(
            self.window, "Save Layer",
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
        tab = self._tab
        path_str, _ = QFileDialog.getOpenFileName(
            self.window, "Open Layer", str(_dirs.DEFAULT_OPEN_DIR),
            "MapCreator Layer (*.mclayer);;All Files (*)",
        )
        if not path_str:
            return
        try:
            with zipfile.ZipFile(Path(path_str), "r") as zf:
                layer = self._read_layer_from_zip(zf, "", tab.scene)
        except Exception as exc:
            error(f"Failed to open layer: {exc}")
            self._status(f"Error opening layer: {exc}", 5000)
            return
        if layer is None:
            self._status("Failed to load layer from file.", 5000)
            return
        tab.layers_list.add_raster_layer(layer)
        if len(tab.layers_list.raster_layers) == 1:
            tab.view.fitInView(layer.item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._status(f"Opened layer: {layer.layer_name}", 3000)

    # ------------------------------------------------------------------
    # Save / open project (multi-tab format, backward-compat with v1)
    # ------------------------------------------------------------------

    def save_project(self) -> None:
        if self._current_project_path is not None:
            self._do_save_project(self._current_project_path)
        else:
            self.save_project_as()

    def save_project_as(self) -> None:
        has_layers = any(t.layers_list.raster_layers for t in self._tabs)
        if not has_layers:
            self._status("No layers to save.", 3000)
            return
        default = (
            self._current_project_path
            if self._current_project_path is not None
            else _dirs.DEFAULT_OPEN_DIR / "project.mcproject"
        )
        path_str, _ = QFileDialog.getSaveFileName(
            self.window, "Save Project As", str(default),
            "MapCreator Project (*.mcproject);;All Files (*)",
        )
        if not path_str:
            return
        save_path = Path(path_str)
        if save_path.suffix.lower() != ".mcproject":
            save_path = save_path.with_suffix(".mcproject")
        self._do_save_project(save_path)

    def _do_save_project(self, save_path: Path) -> None:
        tab_metas = [
            {"name": tab.name, "layer_count": len(tab.layers_list.raster_layers)}
            for tab in self._tabs
        ]
        total = sum(m["layer_count"] for m in tab_metas)
        try:
            with zipfile.ZipFile(save_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("project.json", json.dumps({
                    "world_name": self._world_name,
                    "format_version": 2,
                    "tabs": tab_metas,
                }, indent=2))
                for ti, tab in enumerate(self._tabs):
                    for li, layer in enumerate(tab.layers_list.raster_layers):
                        self._write_layer_to_zip(zf, layer, f"tab_{ti:04d}/layer_{li:04d}/")
        except Exception as exc:
            error(f"Failed to save project: {exc}")
            self._status(f"Error saving project: {exc}", 5000)
            return
        self._current_project_path = save_path
        self._status(f"Saved: {save_path.name} ({total} layers across {len(self._tabs)} tabs)", 3000)

    def open_project(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self.window, "Open Project", str(_dirs.DEFAULT_OPEN_DIR),
            "MapCreator Project (*.mcproject);;All Files (*)",
        )
        if not path_str:
            return
        try:
            with zipfile.ZipFile(Path(path_str), "r") as zf:
                meta = json.loads(zf.read("project.json"))
                fmt = meta.get("format_version", 1)

                # Reset to single empty tab before loading
                assert self._tab_widget is not None
                while len(self._tabs) > 1:
                    self._close_tab(len(self._tabs) - 1)
                self._clear_tab_layers(self._tabs[0])
                self._tabs[0].undo_stack.clear()

                if fmt >= 2:
                    # Multi-tab format
                    tab_metas = meta.get("tabs", [])
                    for ti, tm in enumerate(tab_metas):
                        tab_name = tm.get("name", f"Tab {ti + 1}")
                        layer_count = int(tm.get("layer_count", 0))
                        if ti == 0:
                            tab = self._tabs[0]
                            tab.name = tab_name
                            self._tab_widget.setTabText(0, tab_name)
                        else:
                            tab = self._add_tab(tab_name)
                        self._active_tab_idx = ti
                        for li in range(layer_count - 1, -1, -1):
                            layer = self._read_layer_from_zip(zf, f"tab_{ti:04d}/layer_{li:04d}/", tab.scene)
                            if layer is not None:
                                tab.layers_list.add_raster_layer(layer)
                else:
                    # Legacy single-tab format
                    tab = self._tabs[0]
                    tab.name = "Terrain"
                    self._tab_widget.setTabText(0, "Terrain")
                    layer_count = int(meta.get("layer_count", 0))
                    for i in range(layer_count - 1, -1, -1):
                        layer = self._read_layer_from_zip(zf, f"layer_{i:04d}/", tab.scene)
                        if layer is not None:
                            tab.layers_list.add_raster_layer(layer)

        except Exception as exc:
            error(f"Failed to open project: {exc}")
            self._status(f"Error opening project: {exc}", 5000)
            return

        self._current_project_path = Path(path_str)
        self._update_world_name(meta.get("world_name", ""))

        # Switch to first tab and fit view
        self._tab_widget.setCurrentIndex(0)
        self._active_tab_idx = 0
        first_tab = self._tabs[0]
        if first_tab.layers_list.raster_layers:
            bounds = first_tab.scene.itemsBoundingRect()
            if not bounds.isNull():
                first_tab.view.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)

        total = sum(len(t.layers_list.raster_layers) for t in self._tabs)
        self._status(f"Opened: {Path(path_str).name} ({total} layers, {len(self._tabs)} tabs)", 3000)

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def _undo(self) -> None:
        desc = self._tab.undo_stack.undo()
        self._status(f"Undo: {desc}" if desc else "Nothing to undo.", 1500)

    def _redo(self) -> None:
        desc = self._tab.undo_stack.redo()
        self._status(f"Redo: {desc}" if desc else "Nothing to redo.", 1500)

    def _find_layer_row_in(self, tab: EditTab, layer: RasterLayer) -> int:
        for i, l in enumerate(tab.layers_list.raster_layers):
            if l is layer:
                return i
        return -1

    def _apply_paint_image(self, tab: EditTab, item: QGraphicsPixmapItem, img: QImage) -> None:
        item.setPixmap(QPixmap.fromImage(img))
        if item is tab.paint_item:
            tab.paint_image = img.copy()

    def _push_create_cmd(self, tab: EditTab, layer: RasterLayer, description: str) -> None:
        saved_z = layer.item.zValue()

        def undo() -> None:
            row = self._find_layer_row_in(tab, layer)
            if row < 0:
                return
            if layer.item is tab.paint_item:
                tab.paint_image = None
                tab.paint_item = None
            if layer.is_overlay:
                tab.view.set_overlay_item(None)
            tab.scene.removeItem(layer.item)
            tab.layers_list.remove_layer(row)

        def redo() -> None:
            layer.item.setZValue(saved_z)
            tab.scene.addItem(layer.item)
            tab.layers_list.add_raster_layer(layer)

        tab.undo_stack.push(UndoCommand(undo_fn=undo, redo_fn=redo, description=description))

    def _on_paint_ended(self) -> None:
        tab = self._tab
        if tab.stroke_start_image is None or tab.paint_image is None or tab.paint_item is None:
            tab.stroke_start_image = None
            return
        before = tab.stroke_start_image
        after  = tab.paint_image.copy()
        item   = tab.paint_item
        tab.stroke_start_image = None
        tab.undo_stack.push(UndoCommand(
            undo_fn=lambda: self._apply_paint_image(tab, item, before),
            redo_fn=lambda: self._apply_paint_image(tab, item, after),
            description="Paint stroke",
        ))

    def _on_layer_move_ended(self) -> None:
        tab = self._tab
        if tab.move_start_pos is None:
            return
        row = tab.layers_list.currentRow()
        layers = tab.layers_list.raster_layers
        if not (0 <= row < len(layers)):
            tab.move_start_pos = None
            return
        item = layers[row].item
        start_pos = tab.move_start_pos
        end_pos   = item.pos()
        tab.move_start_pos = None
        if start_pos == end_pos:
            return

        def apply_pos(pos: QPointF) -> None:
            item.setPos(pos)
            if item is tab.paint_item:
                tab.paint_origin = pos
            tab.view.viewport().update()

        tab.undo_stack.push(UndoCommand(
            undo_fn=lambda: apply_pos(start_pos),
            redo_fn=lambda: apply_pos(end_pos),
            description="Move layer",
        ))

    def _on_handle_drag_ended(self) -> None:
        tab = self._tab
        if tab.resize_start_state is None:
            return
        row = tab.layers_list.currentRow()
        layers = tab.layers_list.raster_layers
        if not (0 <= row < len(layers)):
            tab.resize_start_state = None
            return
        item = layers[row].item
        start_pos, start_tf = tab.resize_start_state
        end_pos = item.pos()
        end_tf  = item.transform()
        tab.resize_start_state = None

        def apply(pos: QPointF, tf: QTransform) -> None:
            item.setPos(pos)
            item.setTransform(tf)
            tab.view.viewport().update()

        tab.undo_stack.push(UndoCommand(
            undo_fn=lambda: apply(start_pos, start_tf),
            redo_fn=lambda: apply(end_pos, end_tf),
            description="Resize overlay",
        ))

    def _on_layer_renamed(self, layer: object, old_name: str, new_name: str) -> None:
        tab = self._tab
        rl: RasterLayer = layer  # type: ignore[assignment]

        def apply_name(name: str) -> None:
            rl.layer_name = name
            w = tab.layers_list.get_row_widget(rl)
            if w is not None:
                w.set_name(name)

        tab.undo_stack.push(UndoCommand(
            undo_fn=lambda: apply_name(old_name),
            redo_fn=lambda: apply_name(new_name),
            description=f"Rename to '{new_name}'",
        ))

    def _on_layer_reordered(self, old_order: list, new_order: list) -> None:
        tab = self._tab
        tab.undo_stack.push(UndoCommand(
            undo_fn=lambda: tab.layers_list.restore_order(old_order),
            redo_fn=lambda: tab.layers_list.restore_order(new_order),
            description="Reorder layers",
        ))

    # ------------------------------------------------------------------
    # Public run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        assert self.window is not None
        self.window.show()
        self.qt_app.exec()

    # ------------------------------------------------------------------
    # Layer / overlay creation
    # ------------------------------------------------------------------

    def new_layer_flow(self) -> None:
        self._populate_schema_combo()
        schema = (
            (self.palette_controls.current_schema if self.palette_controls else None)
            or "unknown"
        )
        self._status(f"Opening raster for schema '{schema}'…", 2000)
        self.add_raster_layer_dialog(schema=schema)

    def add_raster_layer_dialog(self, schema: str | None = None) -> None:
        assert self.window is not None
        path_str, _ = QFileDialog.getOpenFileName(
            self.window, "Open Raster", str(_dirs.DEFAULT_OPEN_DIR),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not path_str:
            return
        self.add_raster_layer(Path(path_str), schema=schema)

    def add_raster_layer(self, path: Path, schema: str | None = None) -> None:
        tab = self._tab
        img = QImage(str(path))
        if img.isNull():
            self._status(f"Failed to load image: {path.name}", 5000)
            return
        item = QGraphicsPixmapItem(QPixmap.fromImage(img))
        item.setZValue(len(tab.scene.items()))
        tab.scene.addItem(item)

        layer = RasterLayer(layer_name=path.stem, path=path, item=item, schema=schema or "unknown")
        tab.layers_list.add_raster_layer(layer)

        if len(tab.layers_list.raster_layers) == 1:
            tab.view.fitInView(item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self._push_create_cmd(tab, layer, f"Open '{path.name}'")
        self._status(f"Loaded raster layer: {path.name}", 3000)

    def open_overlay_file(self) -> None:
        tab = self._tab
        path_str, _ = QFileDialog.getOpenFileName(
            self.window, "Open Overlay", str(_dirs.DEFAULT_OPEN_DIR),
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
        item.setZValue(len(tab.scene.items()))
        item.setOpacity(0.5)
        tab.scene.addItem(item)
        layer = RasterLayer(
            layer_name=f"[Ref] {path.stem}", schema="overlay",
            path=path, item=item, is_overlay=True,
        )
        tab.layers_list.add_raster_layer(layer)
        if len(tab.layers_list.raster_layers) == 1:
            tab.view.fitInView(item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._push_create_cmd(tab, layer, f"Open overlay '{path.name}'")
        self._status(f"Opened overlay: {path.name}", 3000)

    # ------------------------------------------------------------------
    # Move / resize mode
    # ------------------------------------------------------------------

    def _on_move_mode_changed(self, active: bool) -> None:
        tab = self._tab
        if active:
            tab.view.set_paint_mode(False)
            tab.view.set_move_mode(True)
        else:
            tab.view.set_move_mode(False)
            has_paint = (
                tab.active_class_color is not None
                or (self.palette_controls is not None and self.palette_controls.is_eraser)
            )
            if has_paint:
                tab.view.set_paint_mode(True)

    def _on_layer_moved(self, dx: float, dy: float) -> None:
        tab = self._tab
        row = tab.layers_list.currentRow()
        layers = tab.layers_list.raster_layers
        if not (0 <= row < len(layers)):
            return
        item = layers[row].item
        if tab.move_start_pos is None:
            tab.move_start_pos = item.pos()
        item.setPos(item.pos() + QPointF(dx, dy))
        if item is tab.paint_item:
            tab.paint_origin = item.pos()

    def _on_handle_dragged(self, handle: int, scene_pos: QPointF) -> None:
        tab = self._tab
        row = tab.layers_list.currentRow()
        layers = tab.layers_list.raster_layers
        if not (0 <= row < len(layers)) or not layers[row].is_overlay:
            return

        item = layers[row].item
        W0 = float(item.pixmap().width())
        H0 = float(item.pixmap().height())
        if W0 <= 0 or H0 <= 0:
            return

        if tab.resize_start_state is None:
            tab.resize_start_state = (item.pos(), item.transform())

        t  = item.transform()
        sx = t.m11() if t.m11() != 0 else 1.0
        sy = t.m22() if t.m22() != 0 else 1.0
        px, py        = item.pos().x(), item.pos().y()
        W,  H         = W0 * sx, H0 * sy
        right, bottom = px + W, py + H
        nx, ny        = scene_pos.x(), scene_pos.y()
        MIN = 10.0

        if handle == 0:    # NW — SE fixed
            nw = max(MIN, right  - nx); nh = max(MIN, bottom - ny)
            item.setPos(right - nw, bottom - nh)
            item.setTransform(QTransform().scale(nw / W0, nh / H0))
        elif handle == 1:  # N  — bottom fixed
            nh = max(MIN, bottom - ny)
            item.setPos(px, bottom - nh)
            item.setTransform(QTransform().scale(sx, nh / H0))
        elif handle == 2:  # NE — SW fixed
            nw = max(MIN, nx - px); nh = max(MIN, bottom - ny)
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
            nw = max(MIN, right  - nx); nh = max(MIN, ny - py)
            item.setPos(right - nw, py)
            item.setTransform(QTransform().scale(nw / W0, nh / H0))
        elif handle == 6:  # S  — top fixed
            nh = max(MIN, ny - py)
            item.setTransform(QTransform().scale(sx, nh / H0))
        elif handle == 7:  # SE — NW fixed
            nw = max(MIN, nx - px); nh = max(MIN, ny - py)
            item.setTransform(QTransform().scale(nw / W0, nh / H0))

    # ------------------------------------------------------------------
    # Canvas toolbar handlers (per-tab)
    # ------------------------------------------------------------------

    def _on_grid_toggled(self, tab: EditTab, checked: bool) -> None:
        assert tab.top_options_bar is not None
        assert tab.grid_size_combo is not None
        tab.top_options_bar.setVisible(checked)
        size = int(tab.grid_size_combo.currentText())
        tab.view.set_grid(checked, size)
        if self.palette_controls is not None:
            self.palette_controls.set_grid_active(checked)

    def _on_grid_size_changed(self, tab: EditTab, text: str) -> None:
        assert tab.grid_toggle_btn is not None
        if not tab.grid_toggle_btn.isChecked():
            return
        try:
            tab.view.set_grid(True, int(text))
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _status(self, msg: str, timeout_ms: int = 2000) -> None:
        if self.window is None:
            return
        status_bar = self.window.statusBar()
        if status_bar is not None:
            status_bar.showMessage(msg, timeout_ms)
            return
        print(f"[STATUS] {msg}")


if __name__ == "__main__":
    MapCreator().run()
