from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QCheckBox, QDockWidget, QGridLayout, QPushButton, QScrollArea, QLabel,
    QVBoxLayout, QHBoxLayout, QToolButton, QSpinBox, QComboBox, QButtonGroup,
)

_NULL_CLASS_ID = -99  # sentinel for "no brush selected"


def _grey_blend(hex_color: str, amount: float = 0.55) -> str:
    """Blend a hex color toward grey by `amount` (0=original, 1=full grey)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    grey = 170
    r = int(r * (1 - amount) + grey * amount)
    g = int(g * (1 - amount) + grey * amount)
    b = int(b * (1 - amount) + grey * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass(frozen=True)
class PaletteClass:
    class_id: int
    palette_name: str
    color_hex: str

def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()

class PaletteButton(QPushButton):
    """A swatch button for selecting a class — color background, name as tooltip."""

    def __init__(self, *, class_id: int, name: str, color_hex: str, parent: QWidget | None = None):
        super().__init__("", parent)
        self.class_id = class_id
        self.name = name
        self.color_hex = color_hex

        self.setToolTip(name)
        self.setCheckable(True)
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color_hex};
                border: 1px solid rgba(0,0,0,0.35);
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border: 2px solid rgba(0,0,0,0.5);
            }}
            QPushButton:checked {{
                border: 3px solid rgba(0,0,0,0.8);
            }}
        """)

class PaletteControls(QObject):
    """
    Controller for the Palette Controls dock area.

    Responsibilities:
    - Find UI widgets inside the dock
    - Populate palette buttons at runtime from schema 'defines'
    - Track selected class id
    - Emit signal when selection changes or brush is cleared
    """

    classSelected = Signal(int)          # emits class_id of selected class
    brushCleared = Signal()              # emits when null/no-brush is selected
    brushSettingsChanged = Signal(int, str)  # (size, shape) whenever spinbox/combo changes
    snapToGridChanged = Signal(bool)     # emits when snap-to-grid checkbox is toggled
    moveModeChanged = Signal(bool)       # emits when move mode button is toggled

    def __init__(self, *, main_window: QWidget):
        super().__init__(main_window)

        self._dock: QDockWidget = self._must_find(main_window, QDockWidget, "layer_palette_controls_dock")
        self._scroll: QScrollArea = self._must_find(self._dock, QScrollArea, "scrollArea")
        self._schema_label: QLabel = self._must_find(self._dock, QLabel, "active_palette_label")
        self._brush_label: QLabel = self._must_find(self._dock, QLabel, "active_brush_label")

        # In the .ui file the QGridLayout lives on gridLayoutWidget (a child of
        # paletteButtonsWidget), not on paletteButtonsWidget itself.
        # gridLayoutWidget also has a hardcoded negative offset geometry (-11, -1)
        # that causes edge buttons to be clipped by the scroll area. Fix by wrapping
        # it in a proper layout on paletteButtonsWidget so it fills the container.
        self._buttons_widget: QWidget = self._must_find(self._dock, QWidget, "gridLayoutWidget")
        layout = self._buttons_widget.layout()
        if not isinstance(layout, QGridLayout):
            raise RuntimeError(
                "gridLayoutWidget must have a QGridLayout. "
                f"Found: {type(layout).__name__ if layout else None}"
            )
        self._grid: QGridLayout = layout

        palette_container: QWidget = self._must_find(self._dock, QWidget, "paletteButtonsWidget")
        if palette_container.layout() is None:
            wrap = QVBoxLayout(palette_container)
            wrap.setContentsMargins(2, 2, 2, 2)
            wrap.setSpacing(0)
            wrap.addWidget(self._buttons_widget)

        self._selected_button: PaletteButton | None = None
        self._null_button: PaletteButton | None = None
        self.selected_class_id: int | None = None

        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(6)
        self._scroll.setWidgetResizable(True)

        self._brush_label.setTextFormat(Qt.TextFormat.RichText)

        # Brush tools
        self._brush_mode_btn: QToolButton = self._must_find(self._dock, QToolButton, "brushModeButton")
        self._eraser_mode_btn: QToolButton = self._must_find(self._dock, QToolButton, "eraserModeButton")
        self._move_mode_btn: QToolButton = self._must_find(self._dock, QToolButton, "moveModeButton")
        self._snap_to_grid_check: QCheckBox = self._must_find(self._dock, QCheckBox, "snapToGridCheckBox")
        self._brush_size_spin: QSpinBox = self._must_find(self._dock, QSpinBox, "brushSizeSpinBox")
        self._brush_shape_combo: QComboBox = self._must_find(self._dock, QComboBox, "brushShapeComboBox")

        self._brush_mode_group = QButtonGroup(self)
        self._brush_mode_group.addButton(self._brush_mode_btn, 0)
        self._brush_mode_group.addButton(self._eraser_mode_btn, 1)
        self._brush_mode_group.addButton(self._move_mode_btn, 2)
        self._brush_mode_group.setExclusive(True)

        self._brush_size_spin.valueChanged.connect(
            lambda v: self.brushSettingsChanged.emit(v, self._brush_shape_combo.currentText())
        )
        self._brush_shape_combo.currentTextChanged.connect(
            lambda s: self.brushSettingsChanged.emit(self._brush_size_spin.value(), s)
        )
        self._snap_to_grid_check.toggled.connect(self.snapToGridChanged)
        self._move_mode_btn.toggled.connect(self.moveModeChanged)

        # Apply filter
        self._apply_swatches_widget: QWidget = self._must_find(self._dock, QWidget, "applySwatchesWidget")
        self._apply_disabled: set[int] = set()

        apply_layout = QHBoxLayout(self._apply_swatches_widget)
        apply_layout.setContentsMargins(0, 0, 0, 0)
        apply_layout.setSpacing(4)
        apply_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._apply_layout = apply_layout

    @staticmethod
    def _must_find(parent: QObject, cls: type, name: str):
        w = parent.findChild(cls, name)
        if w is None:
            raise RuntimeError(f"Could not find {cls.__name__} with objectName='{name}'")
        return w

    @property
    def brush_size(self) -> int:
        return self._brush_size_spin.value()

    @property
    def brush_shape(self) -> str:
        return self._brush_shape_combo.currentText()

    @property
    def is_eraser(self) -> bool:
        return self._eraser_mode_btn.isChecked()

    @property
    def is_move_mode(self) -> bool:
        return self._move_mode_btn.isChecked()

    @property
    def snap_to_grid(self) -> bool:
        return self._snap_to_grid_check.isChecked()

    def set_grid_active(self, active: bool) -> None:
        """Enable or disable the snap-to-grid checkbox based on whether the grid is on."""
        self._snap_to_grid_check.setEnabled(active)
        if not active:
            self._snap_to_grid_check.setChecked(False)

    def set_palette_defines(self, defines: dict[Any, dict[str, Any]], *, columns: int = 6) -> None:
        """
        Build palette buttons from a defines mapping like:
            {0: {name: 'Waterbody', color: '#8BBBEB'}, 1: {...}, -1: {...}}

        Ignores class_id == -1 (Default). Adds a null/no-brush button first.
        """
        classes = []
        for raw_id, meta in defines.items():
            class_id = int(raw_id)
            if class_id == -1:
                continue
            classes.append(
                PaletteClass(
                    class_id=class_id,
                    palette_name=str(meta.get("name", f"Class {class_id}")),
                    color_hex=str(meta.get("color", "#cccccc")),
                )
            )
        classes.sort(key=lambda c: c.class_id)

        self._null_button = None
        _clear_layout(self._grid)
        self._selected_button = None
        self.selected_class_id = None

        # Null / no-brush swatch at position (0, 0)
        null_btn = PaletteButton(
            class_id=_NULL_CLASS_ID, name="No Brush", color_hex="#f0f0f0",
            parent=self._buttons_widget,
        )
        null_btn.setText("∅")
        null_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px dashed rgba(0,0,0,0.25);
                border-radius: 4px;
                color: #999999;
                font-size: 16px;
            }
            QPushButton:hover {
                border: 2px dashed rgba(0,0,0,0.5);
                color: #555555;
            }
            QPushButton:checked {
                border: 3px solid rgba(0,0,0,0.7);
                background-color: #e0e0e0;
                color: #333333;
            }
        """)
        null_btn.clicked.connect(lambda _: self.clear_brush())
        self._grid.addWidget(null_btn, 0, 0)
        self._null_button = null_btn

        # Class swatches — offset index by 1 to account for null at slot 0
        for i, c in enumerate(classes):
            r, col = divmod(i + 1, max(columns, 1))
            btn = PaletteButton(
                class_id=c.class_id, name=c.palette_name, color_hex=c.color_hex,
                parent=self._buttons_widget,
            )
            btn.clicked.connect(lambda _, b=btn: self._on_button_clicked(b))
            self._grid.addWidget(btn, r, col)

        # Start with no brush active
        null_btn.setChecked(True)
        self._update_brush_indicator(None, None)

        self._rebuild_apply_swatches(classes)

    # ------------------------------------------------------------------
    # Apply filter swatches
    # ------------------------------------------------------------------
    def _rebuild_apply_swatches(self, classes: list[PaletteClass]) -> None:
        _clear_layout(self._apply_layout)
        self._apply_disabled.clear()

        for c in classes:
            grey = _grey_blend(c.color_hex)
            btn = QPushButton("", self._apply_swatches_widget)
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.setFixedSize(20, 20)
            btn.setToolTip(c.palette_name)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {c.color_hex};
                    border: 1px solid rgba(0,0,0,0.3);
                    border-radius: 3px;
                }}
                QPushButton:checked {{
                    background-color: {grey};
                    border: 2px solid #444444;
                }}
            """)
            btn.toggled.connect(
                lambda checked, cid=c.class_id: self._on_apply_toggled(cid, checked)
            )
            self._apply_layout.addWidget(btn)

    def _on_apply_toggled(self, class_id: int, checked: bool) -> None:
        if checked:
            self._apply_disabled.add(class_id)
        else:
            self._apply_disabled.discard(class_id)

    def get_disabled_classes(self) -> frozenset[int]:
        return frozenset(self._apply_disabled)

    def set_schema_label(self, schema: str) -> None:
        self._schema_label.setText(f"Active Palette: {schema.capitalize()}")

    def clear_brush(self) -> None:
        if self._selected_button is not None:
            self._selected_button.setChecked(False)
            self._selected_button = None
        self.selected_class_id = None
        if self._null_button is not None:
            self._null_button.setChecked(True)
        self._update_brush_indicator(None, None)
        self.brushCleared.emit()

    def _update_brush_indicator(self, name: str | None, color_hex: str | None) -> None:
        if name is None or color_hex is None:
            self._brush_label.setText("Brush: None")
        else:
            self._brush_label.setText(
                f"Brush: <span style='color: {color_hex};'>■</span> {name}"
            )

    def _on_button_clicked(self, btn: PaletteButton) -> None:
        self._set_selected(btn)
        self.classSelected.emit(btn.class_id)

    def _set_selected(self, btn: PaletteButton) -> None:
        if self._selected_button is btn:
            return
        if self._selected_button is not None:
            self._selected_button.setChecked(False)
        if self._null_button is not None:
            self._null_button.setChecked(False)
        self._selected_button = btn
        btn.setChecked(True)
        self.selected_class_id = btn.class_id
        self._update_brush_indicator(btn.name, btn.color_hex)
