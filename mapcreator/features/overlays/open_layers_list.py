"""
mapcreator.features.overlays.openlayerslist

Defines OpenLayersList, a custom QListWidget for managing map layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import QListWidget, QAbstractItemView, QListWidgetItem
from PySide6.QtGui import QMouseEvent
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtWidgets import QGraphicsPixmapItem
from mapcreator.globals.logutil import info, process_step, error, setting_config, success, warn

@dataclass
class RasterLayer:
    """Simple model so list-widget rows can map to real scene items."""
    layer_name: str 
    schema: str
    path: Path
    item: QGraphicsPixmapItem


class OpenLayersList(QListWidget):
    """
    A QListWidget that manages the list of open layers.
    """
    layerSchemaChanged = Signal(str)  # emits schema name when selected layer changes

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.raster_layers: list[RasterLayer] = []
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.currentRowChanged.connect(self._on_layer_selected)
        self.itemChanged.connect(self._on_item_renamed)

    def add_raster_layer(self, layer: RasterLayer):
        """Adds a raster layer to the list."""
        self.raster_layers.append(layer)
        info(f"Added layer: {layer.layer_name} (schema: {layer.schema}, path: {layer.path})")
        list_item = QListWidgetItem(layer.layer_name)
        list_item.setFlags(list_item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.addItem(list_item)
        self.setCurrentRow(self.count() - 1)

    def _on_item_renamed(self, list_item: QListWidgetItem) -> None:
        row = self.row(list_item)
        if row < 0 or row >= len(self.raster_layers):
            return
        new_name = list_item.text().strip()
        if new_name:
            self.raster_layers[row].layer_name = new_name
            info(f"Layer {row} renamed to: {new_name}")
        else:
            list_item.setText(self.raster_layers[row].layer_name)

    def _on_layer_selected(self, row: int) -> None:
        """Visually highlight the selected layer and notify palette of schema change."""
        info(f"Selected layer index: {row}")
        if row < 0 or row >= len(self.raster_layers):
            return

        for lyr in self.raster_layers:
            lyr.item.setOpacity(1.0)
        self.raster_layers[row].item.setOpacity(0.85)

        self.layerSchemaChanged.emit(self.raster_layers[row].schema)
