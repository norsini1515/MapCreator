from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QDockWidget, QGridLayout, QPushButton, QScrollArea
)

@dataclass(frozen=True)
class PaletteClass:
    class_id: int
    palette_name: str
    color_hex: str

def _clear_layout(layout: QGridLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()

class PaletteButton(QPushButton):
    """A swatch-like button for selecting a class (id/color/name)."""

    def __init__(self, *, class_id: int, name: str, color_hex: str, parent: QWidget | None = None):
        super().__init__(name, parent)
        self.class_id = class_id
        self.color_hex = color_hex

        self.setCheckable(True)
        self.setMinimumHeight(28)
        self.setCursor(Qt.PointingHandCursor)

        # Simple swatch style
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color_hex};
                border: 1px solid rgba(0,0,0,0.35);
                border-radius: 6px;
                padding: 6px 10px;
                text-align: left;
            }}
            QPushButton:checked {{
                border: 2px solid rgba(0,0,0,0.75);
            }}
        """)

class PaletteControls(QObject):
    """
    Controller for the Palette Controls dock area.

    Responsibilities:
    - Find UI widgets inside the dock
    - Populate palette buttons at runtime from schema 'defines'
    - Track selected class id
    - Emit signal when selection changes
    """

    classSelected = Signal(int)  # emits class_id

    def __init__(self, *, main_window: QWidget):
        super().__init__(main_window)

        self._dock: QDockWidget = self._must_find(main_window, QDockWidget, "layer_palette_controls_dock")
        self._scroll: QScrollArea = self._must_find(self._dock, QScrollArea, "scrollArea")
        self._buttons_widget: QWidget = self._must_find(self._dock, QWidget, "paletteButtonsWidget")

        layout = self._buttons_widget.layout()
        if not isinstance(layout, QGridLayout):
            raise RuntimeError(
                "paletteButtonsWidget must have a QGridLayout named 'paletteButtonsGrid'. "
                f"Found: {type(layout).__name__ if layout else None}"
            )
        self._grid: QGridLayout = layout

        self._selected_button: PaletteButton | None = None
        self.selected_class_id: int | None = None

        # Nice defaults for packing
        self._grid.setAlignment(Qt.AlignTop)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(8)

        # Ensure scroll area behaves
        self._scroll.setWidgetResizable(True)

    @staticmethod
    def _must_find(parent: QObject, cls: type, name: str):
        w = parent.findChild(cls, name)
        if w is None:
            raise RuntimeError(f"Could not find {cls.__name__} with objectName='{name}'")
        return w

    def set_palette_defines(self, defines: dict[Any, dict[str, Any]], *, columns: int = 2) -> None:
        """
        Build palette buttons from a defines mapping like:
            {0: {name: 'Waterbody', color: '#8BBBEB'}, 1: {...}, -1: {...}}

        Ignores class_id == -1.
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

        _clear_layout(self._grid)
        self._selected_button = None
        self.selected_class_id = None

        for i, c in enumerate(classes):
            r, col = divmod(i, max(columns, 1))
            btn = PaletteButton(class_id=c.class_id, name=c.palette_name, color_hex=c.color_hex, parent=self._buttons_widget)
            btn.clicked.connect(lambda checked=False, b=btn: self._on_button_clicked(b))
            self._grid.addWidget(btn, r, col)

        # Auto-select first class if available (optional)
        if classes:
            first_btn = self._grid.itemAt(0).widget()
            if isinstance(first_btn, PaletteButton):
                first_btn.setChecked(True)
                self._set_selected(first_btn)

    def _on_button_clicked(self, btn: PaletteButton) -> None:
        self._set_selected(btn)
        self.classSelected.emit(btn.class_id)

    def _set_selected(self, btn: PaletteButton) -> None:
        if self._selected_button is btn:
            return
        if self._selected_button is not None:
            self._selected_button.setChecked(False)
        self._selected_button = btn
        self._selected_button.setChecked(True)
        self.selected_class_id = btn.class_id