"""
Docstring for mapcreator.features.overlays.layer_view

Defines LayerView, a subclass of QGraphicsView with behaviors suitable for a map canvas:
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QEvent, QPointF
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QGraphicsView


class LayerView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._mm_panning = False  # middle-mouse panning state
        
    def wheelEvent(self, event):
        # Zoom under mouse
        if event.angleDelta().y() == 0:
            return

        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15

        current_scale = self.transform().m11()
        if not (0.05 < current_scale * factor < 50):
            return

        self.scale(factor, factor)

    def mousePressEvent(self, event):
        # Enable panning with middle mouse
        if event.button() == Qt.MouseButton.MiddleButton:
            self._mm_panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
            # Fake a LEFT press so ScrollHandDrag engages
            fake = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            super().mousePressEvent(fake)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # While middle is held, keep the "buttons" state as LeftButton for drag
        if self._mm_panning:
            fake = QMouseEvent(
                QEvent.Type.MouseMove,
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            super().mouseMoveEvent(fake)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # Disable panning with middle mouse
        if event.button() == Qt.MouseButton.MiddleButton:
            
            # Fake a LEFT release to end ScrollHandDrag cleanly
            fake = QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                event.modifiers(),
            )
            super().mouseReleaseEvent(fake)
            event.accept()
            return

        super().mouseReleaseEvent(event)
