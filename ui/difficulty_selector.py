"""Seletor de dificuldade da IA."""

import pygame
from typing import Callable, List
from config import ColorsConfig, UIElementConfig
from core.enums import Difficulty, GameMode
from .button import Button


class DifficultySelector:
    """
    Seletor de dificuldade com 3 opções:
    - Fácil
    - Médio
    - Difícil

    Visível apenas quando há IA no jogo.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        on_difficulty_change: Callable[[Difficulty], None]
    ):
        """
        Inicializa o seletor de dificuldade.

        Args:
            x: Posição X do seletor
            y: Posição Y do seletor
            width: Largura dos botões
            on_difficulty_change: Callback quando dificuldade é alterada
        """
        self.x = x
        self.y = y
        self.width = width
        self.on_difficulty_change = on_difficulty_change
        self.current_difficulty = Difficulty.MEDIUM
        self.visible = True  # Visível por padrão

        # Criar botões para cada dificuldade em layout vertical
        self.buttons: List[Button] = []
        difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]

        for i, difficulty in enumerate(difficulties):
            button_y = y + i * UIElementConfig.SELECTOR_BUTTON_HEIGHT  # Sem espaçamento
            button = Button(
                x=x,
                y=button_y,
                width=width,
                height=UIElementConfig.SELECTOR_BUTTON_HEIGHT,
                text=difficulty.value,
                callback=lambda d=difficulty: self._on_button_click(d),
                font_size=UIElementConfig.SELECTOR_FONT_SIZE
            )
            self.buttons.append(button)

        # Marcar botão inicial como selecionado
        self._update_button_states()

    def _on_button_click(self, difficulty: Difficulty) -> None:
        """
        Callback quando um botão é clicado.

        Args:
            difficulty: Dificuldade selecionada
        """
        if difficulty != self.current_difficulty:
            self.current_difficulty = difficulty
            self._update_button_states()
            self.on_difficulty_change(difficulty)

    def _update_button_states(self) -> None:
        """Atualiza estado visual dos botões."""
        difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
        for i, difficulty in enumerate(difficulties):
            # Botão selecionado tem cor diferente
            is_selected = (difficulty == self.current_difficulty)
            if is_selected:
                self.buttons[i].normal_color = ColorsConfig.BUTTON_SELECTED
                self.buttons[i].hover_color = ColorsConfig.BUTTON_SELECTED
            else:
                self.buttons[i].normal_color = ColorsConfig.BUTTON_NORMAL
                self.buttons[i].hover_color = ColorsConfig.BUTTON_HOVER

    def set_visible(self, visible: bool) -> None:
        """
        Define se o seletor está visível.

        Args:
            visible: True para visível, False para oculto
        """
        self.visible = visible

    def update_visibility(self, game_mode: GameMode) -> None:
        """
        Atualiza visibilidade baseado no modo de jogo.

        Args:
            game_mode: Modo de jogo atual
        """
        self.visible = game_mode.has_ai()

    def render(self, screen: pygame.Surface) -> None:
        """
        Renderiza o seletor.

        Args:
            screen: Superfície para renderização
        """
        if not self.visible:
            return

        # Renderizar título mais próximo dos botões
        font = pygame.font.Font(None, UIElementConfig.PANEL_TITLE_FONT_SIZE)
        title_surface = font.render("DIFICULDADE", True, ColorsConfig.TEXT)
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
        if not self.visible:
            return

        for button in self.buttons:
            button.handle_event(event)

    def update(self, mouse_pos: tuple[int, int]) -> None:
        """
        Atualiza estado do seletor.

        Args:
            mouse_pos: Posição do mouse
        """
        if not self.visible:
            return

        # Atualizar estado de hover dos botões
        for button in self.buttons:
            button.update(mouse_pos)
