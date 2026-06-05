"""
mapcreator.features.overlays.open_layers_list

Defines OpenLayersList — a QListWidget where each row is a LayerRowWidget
with an eye-toggle for visibility, an inline-editable name, and an opacity slider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QEvent, Signal, QSize
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QSlider,
    QStackedWidget,
    QToolButton,
    QWidget,
    QGraphicsPixmapItem,
)

from mapcreator.globals.logutil import info


@dataclass
class RasterLayer:
    """Maps a list row to a scene item."""
    layer_name: str
    schema: str
    path: Path
    item: QGraphicsPixmapItem
    is_overlay: bool = False


class LayerRowWidget(QWidget):
    """A single row in the layers panel: [eye] [name / rename-edit] [opacity slider] [pct]"""

    visibilityToggled = Signal(bool)   # True = visible
    opacityChanged    = Signal(int)    # 0–100
    renameRequested   = Signal(str)    # new name

    _ICON_VISIBLE = "◉"
    _ICON_HIDDEN  = "◯"
    _STYLE_VISIBLE = "QToolButton { border: none; color: #4a9eff; font-size: 14px; }"
    _STYLE_HIDDEN  = "QToolButton { border: none; color: #aaaaaa; font-size: 14px; }"

    def __init__(self, name: str, is_overlay: bool = False,
                 initial_opacity: int = 100, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        if is_overlay:
            self.setStyleSheet("background-color: #fff3e0;")

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 3, 4, 3)
        row.setSpacing(4)

        # --- Eye toggle ---
        self._eye_btn = QToolButton()
        self._eye_btn.setCheckable(True)
        self._eye_btn.setChecked(True)
        self._eye_btn.setFixedSize(24, 24)
        self._eye_btn.setText(self._ICON_VISIBLE)
        self._eye_btn.setStyleSheet(self._STYLE_VISIBLE)
        self._eye_btn.setToolTip("Toggle visibility")
        self._eye_btn.toggled.connect(self._on_eye_toggled)
        row.addWidget(self._eye_btn)

        # --- Name: label shown normally, line-edit shown during rename ---
        self._name_stack = QStackedWidget()
        self._name_label = QLabel(name)
        self._name_label.setMinimumWidth(50)
        # Install event filter to catch double-click on the label specifically
        self._name_label.installEventFilter(self)

        self._name_edit = QLineEdit(name)
        self._name_edit.editingFinished.connect(self._commit_rename)

        self._name_stack.addWidget(self._name_label)   # index 0 — display
        self._name_stack.addWidget(self._name_edit)    # index 1 — editing
        self._name_stack.setCurrentIndex(0)
        row.addWidget(self._name_stack, 1)

        # --- Opacity slider ---
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(initial_opacity)
        self._slider.setFixedWidth(72)
        self._slider.setToolTip("Opacity")
        self._slider.valueChanged.connect(self.opacityChanged)
        row.addWidget(self._slider)

        # --- Opacity percent label ---
        self._pct_label = QLabel(f"{initial_opacity}%")
        self._pct_label.setFixedWidth(34)
        self._pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._slider.valueChanged.connect(lambda v: self._pct_label.setText(f"{v}%"))
        row.addWidget(self._pct_label)

    # ------------------------------------------------------------------
    # Event filter — double-click on the name label triggers rename
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event: QEvent) -> bool:
        if obj is self._name_label:
            try:
                if event.type() == QEvent.Type.MouseButtonDblClick:
                    self.start_rename()
                    return True
            except SystemError:
                pass
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Rename logic
    # ------------------------------------------------------------------
    def start_rename(self) -> None:
        self._name_edit.setText(self._name_label.text())
        self._name_edit.selectAll()
        self._name_stack.setCurrentIndex(1)
        self._name_edit.setFocus()

    def _commit_rename(self) -> None:
        if self._name_stack.currentIndex() != 1:
            return
        self._name_stack.setCurrentIndex(0)
        new_name = self._name_edit.text().strip() or self._name_label.text()
        self._name_label.setText(new_name)
        self.renameRequested.emit(new_name)

    # ------------------------------------------------------------------
    # Eye toggle
    # ------------------------------------------------------------------
    def _on_eye_toggled(self, visible: bool) -> None:
        self._eye_btn.setText(self._ICON_VISIBLE if visible else self._ICON_HIDDEN)
        self._eye_btn.setStyleSheet(self._STYLE_VISIBLE if visible else self._STYLE_HIDDEN)
        self.visibilityToggled.emit(visible)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def layer_name(self) -> str:
        return self._name_label.text()


class OpenLayersList(QListWidget):
    """QListWidget where each row is a LayerRowWidget."""

    layerSchemaChanged = Signal(str)  # emits schema of the newly selected layer
    layerSelected = Signal(int)       # emits row index of the newly selected layer
    layerRenamed = Signal()           # emits whenever any layer's name changes
    duplicateRequested = Signal(int)  # emits row index of the layer to duplicate
    deleteRequested = Signal(int)     # emits row index of the layer to delete

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.raster_layers: list[RasterLayer] = []
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.currentRowChanged.connect(self._on_layer_selected)

    # ------------------------------------------------------------------
    # Adding layers
    # ------------------------------------------------------------------
    def add_raster_layer(self, layer: RasterLayer) -> None:
        self.raster_layers.insert(0, layer)
        info(f"Added layer: {layer.layer_name} (schema: {layer.schema}, path: {layer.path})")

        initial_opacity = round(layer.item.opacity() * 100)
        row_widget = LayerRowWidget(layer.layer_name,
                                    is_overlay=layer.is_overlay,
                                    initial_opacity=initial_opacity)

        row_widget.visibilityToggled.connect(
            lambda visible, _layer=layer: _layer.item.setVisible(visible)
        )
        row_widget.opacityChanged.connect(
            lambda value, _layer=layer: _layer.item.setOpacity(value / 100.0)
        )
        row_widget.renameRequested.connect(
            lambda name, _layer=layer: self._on_rename(_layer, name)
        )

        list_item = QListWidgetItem()
        list_item.setSizeHint(QSize(0, 44))
        list_item.setData(Qt.ItemDataRole.UserRole, layer)
        self.insertItem(0, list_item)
        self.setItemWidget(list_item, row_widget)

        self.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Drag-to-reorder
    # ------------------------------------------------------------------
    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        self._sync_after_drop()

    def _sync_after_drop(self) -> None:
        """Rebuild raster_layers from the current visual order and update z-values."""
        new_order: list[RasterLayer] = []
        for i in range(self.count()):
            layer = self.item(i).data(Qt.ItemDataRole.UserRole)
            if layer is not None:
                new_order.append(layer)
        self.raster_layers = new_order
        self._reorder_z_values()

        row = self.currentRow()
        if 0 <= row < len(self.raster_layers):
            self.layerSchemaChanged.emit(self.raster_layers[row].schema)
            self.layerSelected.emit(row)

    # ------------------------------------------------------------------
    # Right-click context menu
    # ------------------------------------------------------------------
    def contextMenuEvent(self, event) -> None:
        item = self.itemAt(event.pos())
        if item is None:
            return
        row = self.row(item)
        if not (0 <= row < len(self.raster_layers)):
            return

        self.setCurrentRow(row)

        menu = QMenu(self)
        act_duplicate = menu.addAction("Duplicate")
        menu.addSeparator()
        act_delete = menu.addAction("Delete")

        chosen = menu.exec(event.globalPos())
        if chosen is act_duplicate:
            self.duplicateRequested.emit(row)
        elif chosen is act_delete:
            self.deleteRequested.emit(row)

    # ------------------------------------------------------------------
    # Removing layers
    # ------------------------------------------------------------------
    def clear_all(self) -> None:
        """Remove all layer rows and data without touching the scene."""
        self.raster_layers.clear()
        self.clear()  # QListWidget.clear()

    def remove_layer(self, row: int) -> None:
        if not (0 <= row < len(self.raster_layers)):
            return
        self.raster_layers.pop(row)
        self.takeItem(row)
        self._reorder_z_values()
        new_row = min(row, self.count() - 1)
        if new_row >= 0:
            self.setCurrentRow(new_row)

    def _reorder_z_values(self) -> None:
        n = len(self.raster_layers)
        for i, layer in enumerate(self.raster_layers):
            layer.item.setZValue(n - i)

    # ------------------------------------------------------------------
    # Row signal handlers
    # ------------------------------------------------------------------
    def _on_rename(self, layer: RasterLayer, name: str) -> None:
        layer.layer_name = name
        info(f"Layer renamed to: {name}")
        self.layerRenamed.emit()

    # ------------------------------------------------------------------
    # Selection → palette sync
    # ------------------------------------------------------------------
    def _on_layer_selected(self, row: int) -> None:
        if 0 <= row < len(self.raster_layers):
            layer = self.raster_layers[row]
            if not layer.is_overlay:
                self.layerSchemaChanged.emit(layer.schema)
            self.layerSelected.emit(row)
