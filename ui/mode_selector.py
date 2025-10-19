"""Seletor de modo de jogo."""

import pygame
from typing import Callable, List
from config import ColorsConfig, UIElementConfig
from core.enums import GameMode
from .button import Button


class ModeSelector:
    """
    Seletor de modo de jogo com 3 opções:
    - Humano vs Humano
    - Humano vs IA
    - IA vs IA
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        on_mode_change: Callable[[GameMode], None]
    ):
        """
        Inicializa o seletor de modo.

        Args:
            x: Posição X do seletor
            y: Posição Y do seletor
            width: Largura dos botões
            on_mode_change: Callback quando modo é alterado
        """
        self.x = x
        self.y = y
        self.width = width
        self.on_mode_change = on_mode_change
        self.current_mode = GameMode.HUMAN_VS_AI

        # Criar botões para cada modo em layout vertical
        self.buttons: List[Button] = []
        modes = [GameMode.HUMAN_VS_HUMAN, GameMode.HUMAN_VS_AI, GameMode.AI_VS_AI]

        for i, mode in enumerate(modes):
            button_y = y + i * UIElementConfig.SELECTOR_BUTTON_HEIGHT  # Sem espaçamento
            button = Button(
                x=x,
                y=button_y,
                width=width,
                height=UIElementConfig.SELECTOR_BUTTON_HEIGHT,
                text=mode.value,  # Nome completo agora
                callback=lambda m=mode: self._on_button_click(m),
                font_size=UIElementConfig.SELECTOR_FONT_SIZE
            )
            self.buttons.append(button)

        # Marcar botão inicial como selecionado
        self._update_button_states()

    def _on_button_click(self, mode: GameMode) -> None:
        """
        Callback quando um botão é clicado.

        Args:
            mode: Modo selecionado
        """
        if mode != self.current_mode:
            self.current_mode = mode
            self._update_button_states()
            self.on_mode_change(mode)

    def _update_button_states(self) -> None:
        """Atualiza estado visual dos botões."""
        modes = [GameMode.HUMAN_VS_HUMAN, GameMode.HUMAN_VS_AI, GameMode.AI_VS_AI]
        for i, mode in enumerate(modes):
            # Botão selecionado tem cor diferente
            is_selected = (mode == self.current_mode)
            if is_selected:
                self.buttons[i].normal_color = ColorsConfig.BUTTON_SELECTED
                self.buttons[i].hover_color = ColorsConfig.BUTTON_SELECTED
            else:
                self.buttons[i].normal_color = ColorsConfig.BUTTON_NORMAL
                self.buttons[i].hover_color = ColorsConfig.BUTTON_HOVER

    def render(self, screen: pygame.Surface) -> None:
        """
        Renderiza o seletor.

        Args:
            screen: Superfície para renderização
        """
        # Renderizar título mais próximo dos botões
        font = pygame.font.Font(None, UIElementConfig.PANEL_TITLE_FONT_SIZE)
        title_surface = font.render("MODO DE JOGO", True, ColorsConfig.TEXT)
        title_rect = title_surface.get_rect(
            topleft=(self.x, self.y - 20)
        )
        screen.blit(title_surface, title_rect)

        # Renderizar botões
        for button in self.buttons:
            button.render(screen)

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Processa eventos.

        Args:
            event: Evento a processar
        """
        for button in self.buttons:
            button.handle_event(event)

    def update(self, mouse_pos: tuple[int, int]) -> None:
        """
        Atualiza estado do seletor.

        Args:
            mouse_pos: Posição do mouse
        """
        # Atualizar estado de hover dos botões
        for button in self.buttons:
            button.update(mouse_pos)
