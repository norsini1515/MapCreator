"""
mapcreator.features.overlays.openlayerslist

Defines OpenLayersList, a custom QListWidget for managing map layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import QListWidget
from PySide6.QtGui import QMouseEvent
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import QGraphicsPixmapItem
from mapcreator.globals.logutil import info, process_step, error, setting_config, success, warn

@dataclass
class RasterLayer:
    """Simple model so list-widget rows can map to real scene items."""
    name: str
    schema: str
    path: Path
    item: QGraphicsPixmapItem


class OpenLayersList(QListWidget):
    """
    A QListWidget that manages the list of open layers.
    """
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.raster_layers: list[RasterLayer] = []
        self.currentRowChanged.connect(self._on_layer_selected)

    def add_raster_layer(self, layer: RasterLayer):
        """Adds a raster layer to the list."""
        self.raster_layers.append(layer)
        info(f"Added layer: {layer.name} (schema: {layer.schema}, path: {layer.path})")
        self.addItem(layer.name)
        self.setCurrentRow(self.count() - 1)

    def _on_layer_selected(self, row: int) -> None:
        """MVP: visually indicate selected layer by changing opacity slightly."""
        info(f"Selected layer index: {row}")
        if row < 0 or row >= len(self.raster_layers):
            return

        # Reset all
        for lyr in self.raster_layers:
            lyr.item.setOpacity(1.0)

        # Highlight selected
        self.raster_layers[row].item.setOpacity(0.85)
