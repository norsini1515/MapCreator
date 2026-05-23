"""
mapcreator.features.overlays.layer_view

Defines LayerView, a subclass of QGraphicsView with behaviors suitable for a map canvas.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QEvent, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QGraphicsView


class LayerView(QGraphicsView):
    paintStroke = Signal(QPointF)  # scene-space position of each paint point

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.viewport().setMouseTracking(True)

        self._mm_panning = False
        self._paint_mode = False
        self._painting = False

        self._brush_size: int = 8
        self._brush_shape: str = "Circle"
        self._cursor_scene_pos: QPointF | None = None

    def set_paint_mode(self, active: bool) -> None:
        self._paint_mode = active
        if active:
            self._painting = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.BlankCursor)
        else:
            self._painting = False
            self._cursor_scene_pos = None
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.unsetCursor()
        self.viewport().update()

    def set_brush(self, size: int, shape: str) -> None:
        self._brush_size = size
        self._brush_shape = shape
        if self._paint_mode:
            self.viewport().update()

    def _to_scene(self, event: QMouseEvent) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    # ------------------------------------------------------------------
    # Brush cursor overlay
    # ------------------------------------------------------------------
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        if not self._paint_mode or self._cursor_scene_pos is None:
            return

        r = float(self._brush_size)
        pos = self._cursor_scene_pos
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # White halo for contrast on dark content
        halo = QPen(QColor(255, 255, 255, 200))
        halo.setWidth(2)
        halo.setCosmetic(True)
        painter.setPen(halo)
        self._draw_brush_shape(painter, pos, r)

        # Black dashed ring on top
        ring = QPen(QColor(0, 0, 0, 180))
        ring.setWidth(1)
        ring.setStyle(Qt.PenStyle.DashLine)
        ring.setCosmetic(True)
        painter.setPen(ring)
        self._draw_brush_shape(painter, pos, r)

    def _draw_brush_shape(self, painter: QPainter, pos: QPointF, r: float) -> None:
        if self._brush_shape == "Square":
            painter.drawRect(QRectF(pos.x() - r, pos.y() - r, r * 2, r * 2))
        else:
            painter.drawEllipse(pos, r, r)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def wheelEvent(self, event):
        if event.angleDelta().y() == 0:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        current_scale = self.transform().m11()
        if not (0.05 < current_scale * factor < 50):
            return
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._mm_panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
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

        if self._paint_mode and event.button() == Qt.MouseButton.LeftButton:
            self._painting = True
            self.paintStroke.emit(self._to_scene(event))
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
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

        if self._paint_mode:
            scene_pos = self._to_scene(event)
            self._cursor_scene_pos = scene_pos
            self.viewport().update()
            if self._painting and (event.buttons() & Qt.MouseButton.LeftButton):
                self.paintStroke.emit(scene_pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._mm_panning = False
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

        if self._paint_mode and event.button() == Qt.MouseButton.LeftButton:
            self._painting = False
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._paint_mode:
                self.paintStroke.emit(self._to_scene(event))
                event.accept()
                return
            self.reset_view()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event):
        self._cursor_scene_pos = None
        self.viewport().update()
        super().leaveEvent(event)

    def reset_view(self):
        scene = self.scene()
        if scene is None:
            return
        rect = scene.itemsBoundingRect()
        if rect.isNull():
            return
        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
