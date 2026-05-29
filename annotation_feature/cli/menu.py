"""Declarative menu action primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MenuAction:
    """A stable menu action independent from its displayed numeric choice."""

    action_id: str
    title: str
    section: str
    handler: Callable[[], None]

    def run(self) -> None:
        self.handler()
