"""Seletor de tipo de avaliador da IA."""

import pygame
from typing import Callable
from config import ColorsConfig, UIElementConfig
from core.enums import GameMode
from .dropdown import Dropdown
import evaluators


class EvaluatorSelector:
    """
    Seletor de avaliador dinâmico que lista todos os avaliadores disponíveis.

    Usa uma caixa de seleção (dropdown) para economizar espaço e melhorar a UX.
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
            width: Largura do dropdown
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

        # Criar dropdown para seleção de avaliador
        # Dropdown fica abaixo do título (título usa 20px de espaço acima)
        dropdown_y = self.y
        self.dropdown = Dropdown(
            x=self.x,
            y=dropdown_y,
            width=self.width,
            height=UIElementConfig.SELECTOR_BUTTON_HEIGHT,
            options=self.evaluator_names,
            default_option=self.current_evaluator,
            on_change=self._on_dropdown_change,
            font_size=UIElementConfig.SELECTOR_FONT_SIZE
        )

    def _on_dropdown_change(self, evaluator_name: str) -> None:
        """
        Callback quando uma opção do dropdown é selecionada.

        Args:
            evaluator_name: Nome do avaliador selecionado
        """
        if evaluator_name != self.current_evaluator:
            self.current_evaluator = evaluator_name
            self.on_evaluator_change(evaluator_name)

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
        Renderiza o seletor (apenas caixa principal do dropdown).

        IMPORTANTE: A parte expandida do dropdown é renderizada separadamente
        via render_expanded() para garantir que flutue sobre outros elementos.

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

        # Renderizar apenas a caixa principal do dropdown
        self.dropdown.render(screen)

    def render_expanded(self, screen: pygame.Surface) -> None:
        """
        Renderiza a parte expandida do dropdown (se estiver expandido).

        Este método deve ser chamado por ÚLTIMO no ciclo de renderização
        para garantir que flutue sobre todos os outros elementos.

        Args:
            screen: Superfície para renderização
        """
        if not self.visible:
            return

        # Renderizar a lista expandida do dropdown (se estiver aberto)
        self.dropdown.render_expanded(screen)

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Processa eventos.

        Args:
            event: Evento a processar
        """
        if not self.visible:
            return

        self.dropdown.handle_event(event)

    def update(self, mouse_pos: tuple[int, int]) -> None:
        """
        Atualiza estado do seletor.

        Args:
            mouse_pos: Posição do mouse
        """
        if not self.visible:
            return

        self.dropdown.update(mouse_pos)
