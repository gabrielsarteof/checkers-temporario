"""Configurações de elementos de UI."""

from dataclasses import dataclass


@dataclass
class UIElementConfig:
    """Configurações para elementos de UI (botões, seletores, etc)."""

    # Seletores de modo e dificuldade (3 botões horizontais)
    SELECTOR_BUTTON_WIDTH: int = 140
    SELECTOR_BUTTON_HEIGHT: int = 35
    SELECTOR_SPACING: int = 5  # Espaço entre botões

    # Botões de controle
    CONTROL_BUTTON_WIDTH: int = 180
    CONTROL_BUTTON_HEIGHT: int = 45
    CONTROL_BUTTON_SPACING: int = 15

    # Fontes
    SELECTOR_FONT_SIZE: int = 18
    CONTROL_FONT_SIZE: int = 22
    PANEL_TITLE_FONT_SIZE: int = 20
    PANEL_TEXT_FONT_SIZE: int = 16

    # Painéis de informação
    PANEL_PADDING: int = 15
    PANEL_SPACING: int = 10
