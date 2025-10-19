"""Painel de controle do jogo."""

import pygame
from typing import Callable, Optional
from config import WindowConfig, BoardConfig, ColorsConfig, UIElementConfig
from core.enums import PlayerColor, GameStatus
from .button import Button


class ControlPanel:
    """
    Painel de controle simplificado.

    Exibe:
    - Botão "Reiniciar"
    - Informações do turno e status do jogo
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        on_reset: Callable[[], None]
    ):
        """
        Inicializa o painel de controle.

        Args:
            x: Posição X do painel
            y: Posição Y do painel
            width: Largura do painel
            on_reset: Callback para quando clicar em "Reiniciar"
        """
        self.panel_x = x
        self.panel_y = y
        self.panel_width = width
        self.panel_height = UIElementConfig.SELECTOR_BUTTON_HEIGHT + 35

        # Fontes
        self.title_font = pygame.font.Font(None, UIElementConfig.PANEL_TITLE_FONT_SIZE)
        self.text_font = pygame.font.Font(None, UIElementConfig.PANEL_TEXT_FONT_SIZE)

        # Botão de reiniciar - mesmas dimensões dos seletores
        self.reset_button = Button(
            x=x,
            y=y,
            width=width,
            height=UIElementConfig.SELECTOR_BUTTON_HEIGHT,
            text="Reiniciar",
            callback=on_reset,
            font_size=UIElementConfig.SELECTOR_FONT_SIZE
        )

        # Estado atual
        self.current_player = PlayerColor.RED
        self.game_status = GameStatus.PLAYING

    def render(self, screen: pygame.Surface) -> None:
        """
        Renderiza o painel de controle.

        Args:
            screen: Superfície para renderização
        """
        # Renderizar título mais próximo do botão (igual aos outros)
        title_surface = self.title_font.render("CONTROLE", True, ColorsConfig.TEXT)
        title_rect = title_surface.get_rect(
            topleft=(self.panel_x, self.panel_y - 20)
        )
        screen.blit(title_surface, title_rect)

        # Renderizar botão
        self.reset_button.render(screen)

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Processa eventos do pygame.

        Args:
            event: Evento a processar
        """
        self.reset_button.handle_event(event)

    def update(self, mouse_pos: tuple[int, int]) -> None:
        """
        Atualiza estado do painel.

        Args:
            mouse_pos: Posição do mouse
        """
        self.reset_button.update(mouse_pos)