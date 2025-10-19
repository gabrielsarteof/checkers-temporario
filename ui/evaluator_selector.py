"""Seletor de tipo de avaliador da IA."""

import pygame
from typing import Callable, List
from config import ColorsConfig, UIElementConfig
from core.enums import GameMode
from .button import Button
import evaluators


class EvaluatorSelector:
    """
    Seletor de avaliador dinâmico que lista todos os avaliadores disponíveis.

    Visível apenas quando há IA no jogo.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        title: str,
        on_evaluator_change: Callable[[str], None],
        default_evaluator: str = None
    ):
        """
        Inicializa o seletor de avaliador.

        Args:
            x: Posição X do seletor
            y: Posição Y do seletor
            width: Largura dos botões
            title: Título do seletor (ex: "AVALIADOR RED", "AVALIADOR BLACK")
            on_evaluator_change: Callback quando avaliador é alterado (recebe nome do avaliador)
            default_evaluator: Nome do avaliador padrão
        """
        self.x = x
        self.y = y
        self.width = width
        self.title = title
        self.on_evaluator_change = on_evaluator_change
        self.visible = True  # Visível por padrão

        # Obter lista de avaliadores disponíveis
        self.evaluator_names = evaluators.get_evaluator_names()

        # Definir avaliador atual
        if default_evaluator and default_evaluator in self.evaluator_names:
            self.current_evaluator = default_evaluator
        else:
            self.current_evaluator = evaluators.get_default_evaluator_name()

        # Criar botões para cada avaliador em layout vertical
        self.buttons: List[Button] = []
        self._create_buttons()

    def _create_buttons(self) -> None:
        """Cria botões para todos os avaliadores disponíveis."""
        self.buttons.clear()

        for i, evaluator_name in enumerate(self.evaluator_names):
            button_y = self.y + i * UIElementConfig.SELECTOR_BUTTON_HEIGHT
            button = Button(
                x=self.x,
                y=button_y,
                width=self.width,
                height=UIElementConfig.SELECTOR_BUTTON_HEIGHT,
                text=evaluator_name,
                callback=lambda e=evaluator_name: self._on_button_click(e),
                font_size=UIElementConfig.SELECTOR_FONT_SIZE
            )
            self.buttons.append(button)

        # Marcar botão inicial como selecionado
        self._update_button_states()

    def _on_button_click(self, evaluator_name: str) -> None:
        """
        Callback quando um botão é clicado.

        Args:
            evaluator_name: Nome do avaliador selecionado
        """
        if evaluator_name != self.current_evaluator:
            self.current_evaluator = evaluator_name
            self._update_button_states()
            self.on_evaluator_change(evaluator_name)

    def _update_button_states(self) -> None:
        """Atualiza estado visual dos botões."""
        for i, evaluator_name in enumerate(self.evaluator_names):
            # Botão selecionado tem cor diferente
            is_selected = (evaluator_name == self.current_evaluator)
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

        # Renderizar título
        font = pygame.font.Font(None, UIElementConfig.PANEL_TITLE_FONT_SIZE)
        title_surface = font.render(self.title, True, ColorsConfig.TEXT)
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
