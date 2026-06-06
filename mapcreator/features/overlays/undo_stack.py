from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class UndoCommand:
    undo_fn: Callable[[], None]
    redo_fn: Callable[[], None]
    description: str = ""


class UndoStack:
    """Paired undo/redo stack — keeps at most MAX commands."""

    MAX = 50

    def __init__(self) -> None:
        self._cmds: list[UndoCommand] = []
        self._pos: int = 0  # index of next push

    def push(self, cmd: UndoCommand) -> None:
        del self._cmds[self._pos:]
        self._cmds.append(cmd)
        if len(self._cmds) > self.MAX:
            self._cmds.pop(0)
        else:
            self._pos = len(self._cmds)

    def undo(self) -> str | None:
        """Execute undo; returns description or None if stack is empty."""
        if self._pos == 0:
            return None
        self._pos -= 1
        cmd = self._cmds[self._pos]
        cmd.undo_fn()
        return cmd.description

    def redo(self) -> str | None:
        """Execute redo; returns description or None if at top."""
        if self._pos >= len(self._cmds):
            return None
        cmd = self._cmds[self._pos]
        cmd.redo_fn()
        self._pos += 1
        return cmd.description

    @property
    def can_undo(self) -> bool:
        return self._pos > 0

    @property
    def can_redo(self) -> bool:
        return self._pos < len(self._cmds)

    def clear(self) -> None:
        self._cmds.clear()
        self._pos = 0
