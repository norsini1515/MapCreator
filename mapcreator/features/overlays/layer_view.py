"""
mapcreator.features.overlays.layer_view

Defines LayerView, a subclass of QGraphicsView with behaviors suitable for a map canvas.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QEvent, QPointF, QRectF, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsView


class LayerView(QGraphicsView):
    paintStroke   = Signal(QPointF)      # scene-space position of each paint point
    cursorMoved   = Signal(QPointF)      # scene-space cursor position on every move
    zoomChanged   = Signal(float)        # current scale factor after any zoom change
    cursorLeft    = Signal()             # cursor has left the viewport
    layerMoved    = Signal(float, float) # (dx, dy) scene-space delta while dragging in move mode
    handleDragged = Signal(int, QPointF) # (handle_index, scene_pos) while resizing an overlay
    paintEnded    = Signal()             # left-button released after a paint/erase stroke
    layerMoveEnded  = Signal()           # left-button released after a move drag
    handleDragEnded = Signal()           # left-button released after a resize drag

    # Handle layout: 0=NW 1=N 2=NE 3=W 4=E 5=SW 6=S 7=SE
    _HANDLE_SIZE = 8  # screen pixels (half-size used for hit detection)
    _HANDLE_CURSORS = [
        Qt.CursorShape.SizeFDiagCursor,  # NW
        Qt.CursorShape.SizeVerCursor,    # N
        Qt.CursorShape.SizeBDiagCursor,  # NE
        Qt.CursorShape.SizeHorCursor,    # W
        Qt.CursorShape.SizeHorCursor,    # E
        Qt.CursorShape.SizeBDiagCursor,  # SW
        Qt.CursorShape.SizeVerCursor,    # S
        Qt.CursorShape.SizeFDiagCursor,  # SE
    ]

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

        self._grid_enabled: bool = False
        self._grid_count: int = 100   # number of cells along each axis
        self._snap_mode: bool = False

        self._move_mode: bool = False
        self._move_last_pos: QPointF | None = None

        self._overlay_item: QGraphicsPixmapItem | None = None
        self._resize_handle: int | None = None

    def set_paint_mode(self, active: bool) -> None:
        self._paint_mode = active
        if active:
            self._painting = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(
                Qt.CursorShape.CrossCursor if self._snap_mode
                else Qt.CursorShape.BlankCursor
            )
        else:
            self._painting = False
            self._cursor_scene_pos = None
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.unsetCursor()
        self.viewport().update()

    def set_snap_mode(self, active: bool) -> None:
        self._snap_mode = active
        if self._paint_mode:
            self.setCursor(
                Qt.CursorShape.CrossCursor if active
                else Qt.CursorShape.BlankCursor
            )
        self.viewport().update()

    def set_move_mode(self, active: bool) -> None:
        self._move_mode = active
        self._move_last_pos = None
        self._resize_handle = None
        if active:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            if not self._paint_mode:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.unsetCursor()
        self.viewport().update()

    def set_overlay_item(self, item: QGraphicsPixmapItem | None) -> None:
        self._overlay_item = item
        self._resize_handle = None
        self.viewport().update()

    def _restore_cursor(self) -> None:
        """Restore the correct cursor after middle-mouse panning ends."""
        if self._move_mode:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif self._paint_mode:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(
                Qt.CursorShape.CrossCursor if self._snap_mode
                else Qt.CursorShape.BlankCursor
            )
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.unsetCursor()

    def set_brush(self, size: int, shape: str) -> None:
        self._brush_size = size
        self._brush_shape = shape
        if self._paint_mode:
            self.viewport().update()

    def set_grid(self, enabled: bool, count: int) -> None:
        self._grid_enabled = enabled
        self._grid_count = max(1, count)
        sc = self.scene()
        if sc is not None:
            sc.update()
        else:
            self.viewport().update()

    def _to_scene(self, event: QMouseEvent) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    # ------------------------------------------------------------------
    # Foreground: grid overlay + snap highlight / brush cursor
    # ------------------------------------------------------------------
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)

        # --- Grid overlay (above all layers, below cursor overlay) ---
        if self._grid_enabled and self._grid_count > 0:
            sc = self.scene()
            if sc is not None:
                bounds = sc.itemsBoundingRect()
                if not bounds.isNull():
                    self._draw_grid(painter, rect, bounds)

        # --- Overlay resize handles ---
        if self._move_mode and self._overlay_item is not None:
            self._draw_overlay_handles(painter)

        if not self._paint_mode or self._cursor_scene_pos is None:
            return

        # --- Snap mode: highlight the hovered grid cell ---
        if self._snap_mode:
            sc = self.scene()
            if sc is not None:
                bounds = sc.itemsBoundingRect()
                if not bounds.isNull():
                    self._draw_snap_highlight(painter, bounds)
            return  # no brush-circle overlay in snap mode

        # --- Normal brush cursor ---
        r = float(self._brush_size)
        pos = self._cursor_scene_pos
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        halo = QPen(QColor(255, 255, 255, 200))
        halo.setWidth(2)
        halo.setCosmetic(True)
        painter.setPen(halo)
        self._draw_brush_shape(painter, pos, r)

        ring = QPen(QColor(0, 0, 0, 180))
        ring.setWidth(1)
        ring.setStyle(Qt.PenStyle.DashLine)
        ring.setCosmetic(True)
        painter.setPen(ring)
        self._draw_brush_shape(painter, pos, r)

    def _draw_grid(self, painter: QPainter, rect: QRectF, bounds: QRectF) -> None:
        count = self._grid_count
        cell_w = bounds.width()  / count
        cell_h = bounds.height() / count
        if cell_w < 0.5 or cell_h < 0.5:
            return

        pen = QPen(QColor(80, 80, 160, 90))
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        clip = rect.intersected(bounds)
        if clip.isEmpty():
            return

        top, bot   = clip.top(),  clip.bottom()
        left, right = clip.left(), clip.right()

        # Vertical lines — only indices that fall within the clip
        i0 = max(0,     int((left  - bounds.left()) / cell_w))
        i1 = min(count, int((right - bounds.left()) / cell_w) + 1)
        for i in range(i0, i1 + 1):
            x = bounds.left() + i * cell_w
            if left <= x <= right:
                painter.drawLine(QPointF(x, top), QPointF(x, bot))

        # Horizontal lines
        j0 = max(0,     int((top - bounds.top()) / cell_h))
        j1 = min(count, int((bot - bounds.top()) / cell_h) + 1)
        for j in range(j0, j1 + 1):
            y = bounds.top() + j * cell_h
            if top <= y <= bot:
                painter.drawLine(QPointF(left, y), QPointF(right, y))

    def _draw_brush_shape(self, painter: QPainter, pos: QPointF, r: float) -> None:
        if self._brush_shape == "Square":
            painter.drawRect(QRectF(pos.x() - r, pos.y() - r, r * 2, r * 2))
        else:
            painter.drawEllipse(pos, r, r)

    def _draw_snap_highlight(self, painter: QPainter, bounds: QRectF) -> None:
        count = self._grid_count
        cell_w = bounds.width()  / count
        cell_h = bounds.height() / count
        if cell_w < 0.5 or cell_h < 0.5:
            return

        pos = self._cursor_scene_pos
        col = max(0, min(count - 1, int((pos.x() - bounds.left()) / cell_w)))
        row = max(0, min(count - 1, int((pos.y() - bounds.top())  / cell_h)))

        cell = QRectF(
            bounds.left() + col * cell_w,
            bounds.top()  + row * cell_h,
            cell_w, cell_h,
        )

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(QColor(100, 150, 255, 60))
        border = QPen(QColor(60, 120, 255, 220))
        border.setCosmetic(True)
        border.setWidthF(1.5)
        painter.setPen(border)
        painter.drawRect(cell)

    def _get_handle_positions(self) -> list[QPointF]:
        """8 handle positions in scene space: NW N NE W E SW S SE."""
        r = self._overlay_item.sceneBoundingRect()
        cx, cy = r.center().x(), r.center().y()
        return [
            QPointF(r.left(),  r.top()),
            QPointF(cx,        r.top()),
            QPointF(r.right(), r.top()),
            QPointF(r.left(),  cy),
            QPointF(r.right(), cy),
            QPointF(r.left(),  r.bottom()),
            QPointF(cx,        r.bottom()),
            QPointF(r.right(), r.bottom()),
        ]

    def _hit_handle(self, vp_x: int, vp_y: int) -> int | None:
        """Return handle index if (vp_x, vp_y) is within _HANDLE_SIZE pixels of one."""
        hs = self._HANDLE_SIZE
        for i, scene_pt in enumerate(self._get_handle_positions()):
            vp = self.mapFromScene(scene_pt)
            if abs(vp.x() - vp_x) <= hs and abs(vp.y() - vp_y) <= hs:
                return i
        return None

    def _draw_overlay_handles(self, painter: QPainter) -> None:
        r = self._overlay_item.sceneBoundingRect()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Dashed bounding rect
        border = QPen(QColor(0, 120, 255, 180))
        border.setCosmetic(True)
        border.setStyle(Qt.PenStyle.DashLine)
        border.setWidthF(1.0)
        painter.setPen(border)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(r)

        # Handle squares — fixed screen size regardless of zoom
        scale = self.transform().m11()
        hs = self._HANDLE_SIZE / scale  # half-size in scene units
        handle_pen = QPen(QColor(0, 100, 220, 220))
        handle_pen.setCosmetic(True)
        handle_pen.setWidthF(1.0)
        painter.setPen(handle_pen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 210)))
        for pt in self._get_handle_positions():
            painter.drawRect(QRectF(pt.x() - hs, pt.y() - hs, hs * 2, hs * 2))

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def wheelEvent(self, event):
        if event.angleDelta().y() == 0:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            current_scale = self.transform().m11()
            if not (0.05 < current_scale * factor < 50):
                return
            self.scale(factor, factor)
            self.zoomChanged.emit(self.transform().m11())
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            delta = event.angleDelta().y()
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - delta // 3)
        else:
            super().wheelEvent(event)

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

        if self._move_mode and event.button() == Qt.MouseButton.LeftButton:
            vp = event.position().toPoint()
            if self._overlay_item is not None:
                h = self._hit_handle(vp.x(), vp.y())
                if h is not None:
                    self._resize_handle = h
                    event.accept()
                    return
            self._move_last_pos = self._to_scene(event)
            event.accept()
            return

        if self._paint_mode and event.button() == Qt.MouseButton.LeftButton:
            self._painting = True
            self.paintStroke.emit(self._to_scene(event))
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self._to_scene(event)
        self.cursorMoved.emit(scene_pos)

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

        if self._move_mode:
            if event.buttons() & Qt.MouseButton.LeftButton:
                if self._resize_handle is not None:
                    self.handleDragged.emit(self._resize_handle, scene_pos)
                elif self._move_last_pos is not None:
                    delta = scene_pos - self._move_last_pos
                    self._move_last_pos = scene_pos
                    self.layerMoved.emit(delta.x(), delta.y())
            else:
                # Hovering — show resize cursor near handles, move cursor otherwise
                if self._overlay_item is not None:
                    vp = event.position().toPoint()
                    h = self._hit_handle(vp.x(), vp.y())
                    self.setCursor(
                        self._HANDLE_CURSORS[h] if h is not None
                        else Qt.CursorShape.SizeAllCursor
                    )
            self.viewport().update()
            event.accept()
            return

        if self._paint_mode:
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
            self._restore_cursor()
            event.accept()
            return

        if self._move_mode and event.button() == Qt.MouseButton.LeftButton:
            if self._resize_handle is not None:
                self.handleDragEnded.emit()
            elif self._move_last_pos is not None:
                self.layerMoveEnded.emit()
            self._move_last_pos = None
            self._resize_handle = None
            event.accept()
            return

        if self._paint_mode and event.button() == Qt.MouseButton.LeftButton:
            if self._painting:
                self.paintEnded.emit()
            self._painting = False
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._move_mode:
                event.accept()
                return
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
        self.cursorLeft.emit()
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
        self.zoomChanged.emit(self.transform().m11())
