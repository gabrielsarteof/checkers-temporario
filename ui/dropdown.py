"""Componente de caixa de seleção (dropdown/combobox)."""

import pygame
from typing import Callable, List, Optional
from config import ColorsConfig, UIElementConfig


class Dropdown:
    """
    Caixa de seleção dropdown com lista de opções flutuante.

    Segue as melhores práticas de UI/UX:
    - Mostra a opção atual em uma caixa clicável com estilo idêntico aos botões
    - Expande para mostrar todas as opções quando clicado (flutua sobre outros elementos)
    - Fecha automaticamente quando uma opção é selecionada ou clique fora
    - Suporta hover visual para melhor feedback
    - Usa exatamente o mesmo estilo visual dos botões existentes
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        options: List[str],
        default_option: Optional[str] = None,
        on_change: Optional[Callable[[str], None]] = None,
        font_size: int = UIElementConfig.SELECTOR_FONT_SIZE
    ):
        """
        Inicializa o dropdown.

        Args:
            x: Posição X do dropdown
            y: Posição Y do dropdown
            width: Largura do dropdown
            height: Altura de cada item
            options: Lista de opções disponíveis
            default_option: Opção inicial selecionada (None = primeira opção)
            on_change: Callback quando opção é alterada (recebe string da opção)
            font_size: Tamanho da fonte
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.options = options
        self.on_change = on_change
        self.font_size = font_size

        # Estado do dropdown
        self.is_expanded = False
        self.hovered_index = -1  # Índice da opção com hover (-1 = nenhuma)
        self.main_hovered = False  # Se a caixa principal está com hover

        # Opção atual selecionada
        if default_option and default_option in options:
            self.selected_option = default_option
            self.selected_index = options.index(default_option)
        else:
            self.selected_option = options[0] if options else ""
            self.selected_index = 0

        # Configurações visuais (idênticas aos botões)
        self.normal_color = ColorsConfig.BUTTON_NORMAL
        self.hover_color = ColorsConfig.BUTTON_HOVER
        self.selected_color = ColorsConfig.BUTTON_SELECTED
        self.border_color = ColorsConfig.BUTTON_BORDER
        self.text_color = ColorsConfig.BUTTON_TEXT

        # Fonte
        self.font = pygame.font.Font(None, self.font_size)

        # Seta para indicar dropdown
        self.arrow_size = 6

    def get_main_rect(self) -> pygame.Rect:
        """Retorna o retângulo da caixa principal (mostra opção selecionada)."""
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def get_option_rect(self, index: int) -> pygame.Rect:
        """
        Retorna o retângulo de uma opção específica na lista expandida.

        Args:
            index: Índice da opção

        Returns:
            Retângulo da opção
        """
        option_y = self.y + self.height + (index * self.height)
        return pygame.Rect(self.x, option_y, self.width, self.height)

    def get_expanded_rect(self) -> pygame.Rect:
        """Retorna o retângulo completo quando expandido (incluindo todas as opções)."""
        total_height = self.height * (len(self.options) + 1)
        return pygame.Rect(self.x, self.y, self.width, total_height)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Processa eventos de mouse.

        Args:
            event: Evento pygame

        Returns:
            True se o evento foi processado, False caso contrário
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            # Clique na caixa principal - toggle expand/collapse
            if self.get_main_rect().collidepoint(mouse_pos):
                self.is_expanded = not self.is_expanded
                return True

            # Se expandido, verifica clique nas opções
            if self.is_expanded:
                for i, option in enumerate(self.options):
                    if self.get_option_rect(i).collidepoint(mouse_pos):
                        # Opção selecionada
                        if self.selected_option != option:
                            self.selected_option = option
                            self.selected_index = i
                            if self.on_change:
                                self.on_change(option)
                        self.is_expanded = False
                        return True

                # Clique fora - fecha dropdown
                self.is_expanded = False
                return False

        return False

    def update(self, mouse_pos: tuple[int, int]) -> None:
        """
        Atualiza estado do dropdown baseado na posição do mouse.

        Args:
            mouse_pos: Posição (x, y) do mouse
        """
        # Atualiza hover na caixa principal
        self.main_hovered = self.get_main_rect().collidepoint(mouse_pos)

        if not self.is_expanded:
            self.hovered_index = -1
            return

        # Atualiza hover nas opções
        self.hovered_index = -1
        for i in range(len(self.options)):
            if self.get_option_rect(i).collidepoint(mouse_pos):
                self.hovered_index = i
                break

    def render(self, screen: pygame.Surface) -> None:
        """
        Renderiza apenas a caixa principal do dropdown.

        IMPORTANTE: A lista expandida deve ser renderizada separadamente usando
        render_expanded() para garantir que flutue sobre outros elementos.

        Args:
            screen: Superfície pygame para desenhar
        """
        # Renderiza apenas a caixa principal
        self._render_main_box(screen)

    def render_expanded(self, screen: pygame.Surface) -> None:
        """
        Renderiza a lista de opções expandida (se estiver expandido).

        Este método deve ser chamado por ÚLTIMO no ciclo de renderização
        para garantir que a lista flutue sobre todos os outros elementos.

        Args:
            screen: Superfície pygame para desenhar
        """
        if self.is_expanded:
            self._render_options(screen)

    def _render_main_box(self, screen: pygame.Surface) -> None:
        """Renderiza a caixa principal com a opção selecionada."""
        main_rect = self.get_main_rect()

        # Determina cor de fundo baseada no estado (igual aos botões)
        if self.main_hovered:
            bg_color = self.hover_color
        else:
            bg_color = self.normal_color

        # Fundo
        pygame.draw.rect(screen, bg_color, main_rect)

        # Borda (mesma espessura dos botões: 2px)
        pygame.draw.rect(screen, self.border_color, main_rect, 2)

        # Texto da opção selecionada (centralizado como nos botões)
        text_surface = self.font.render(self.selected_option, True, self.text_color)

        # Calcular posição do texto considerando espaço para a seta
        arrow_space = self.arrow_size * 2 + 10
        text_rect = text_surface.get_rect(
            centery=main_rect.centery,
            centerx=main_rect.centerx - arrow_space // 4
        )
        screen.blit(text_surface, text_rect)

        # Seta para baixo/cima indicando que é um dropdown
        arrow_x = main_rect.right - self.arrow_size - 8
        arrow_y = main_rect.centery
        self._draw_arrow(screen, arrow_x, arrow_y, down=not self.is_expanded)

    def _render_options(self, screen: pygame.Surface) -> None:
        """Renderiza a lista de opções quando expandido (flutuando sobre os elementos)."""
        # Desenhar uma sombra sutil para dar efeito de profundidade
        shadow_rect = pygame.Rect(
            self.x + 3,
            self.y + self.height + 3,
            self.width,
            len(self.options) * self.height
        )
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height))
        shadow_surface.set_alpha(50)  # Transparência para a sombra
        shadow_surface.fill((0, 0, 0))
        screen.blit(shadow_surface, shadow_rect)

        # Renderizar cada opção
        for i, option in enumerate(self.options):
            option_rect = self.get_option_rect(i)

            # Determina cor de fundo (igual aos botões)
            if i == self.hovered_index:
                bg_color = self.hover_color
            else:
                bg_color = self.normal_color

            # Fundo
            pygame.draw.rect(screen, bg_color, option_rect)

            # Borda (mesma espessura dos botões: 2px)
            pygame.draw.rect(screen, self.border_color, option_rect, 2)

            # Texto (centralizado como nos botões)
            text_surface = self.font.render(option, True, self.text_color)
            text_rect = text_surface.get_rect(center=option_rect.center)
            screen.blit(text_surface, text_rect)

    def _draw_arrow(self, screen: pygame.Surface, x: int, y: int, down: bool = True) -> None:
        """
        Desenha uma seta indicadora.

        Args:
            screen: Superfície pygame
            x: Posição X do centro da seta
            y: Posição Y do centro da seta
            down: True para seta para baixo, False para cima
        """
        half_size = self.arrow_size // 2

        if down:
            # Seta para baixo: ▼
            points = [
                (x, y + half_size),           # Ponta inferior
                (x - half_size, y - half_size),  # Canto superior esquerdo
                (x + half_size, y - half_size)   # Canto superior direito
            ]
        else:
            # Seta para cima: ▲
            points = [
                (x, y - half_size),           # Ponta superior
                (x - half_size, y + half_size),  # Canto inferior esquerdo
                (x + half_size, y + half_size)   # Canto inferior direito
            ]

        pygame.draw.polygon(screen, self.text_color, points)

    def set_selected(self, option: str) -> None:
        """
        Define a opção selecionada programaticamente.

        Args:
            option: Opção a ser selecionada
        """
        if option in self.options:
            self.selected_option = option
            self.selected_index = self.options.index(option)

    def get_selected(self) -> str:
        """
        Retorna a opção atualmente selecionada.

        Returns:
            String da opção selecionada
        """
        return self.selected_option
