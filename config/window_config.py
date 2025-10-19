"""Configurações da janela principal."""

from dataclasses import dataclass


@dataclass
class WindowConfig:
    """Configurações da janela principal."""
    WIDTH: int = 850
    HEIGHT: int = 680
    TITLE: str = "Jogo de Damas - IA vs IA"
    FPS: int = 60