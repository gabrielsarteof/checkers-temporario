"""Painel rolável para componentes UI."""

import pygame
from typing import List, Tuple, Callable, Any


class ScrollablePanel:
    """
    Painel rolável que renderiza componentes com scroll vertical.

    Usa uma superfície virtual grande e exibe apenas a janela visível,
    aplicando offset de scroll nas coordenadas do mouse para eventos.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        scroll_speed: int = 30
    ):
        """
        Inicializa o painel rolável.

        Args:
            x: Posição X do painel na tela
            y: Posição Y do painel na tela
            width: Largura visível do painel
            height: Altura visível do painel
            scroll_speed: Velocidade do scroll em pixels
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.scroll_speed = scroll_speed

        # Offset do scroll (quantos pixels foram rolados)
        self.scroll_offset = 0

        # Lista de componentes: (componente, render_func, update_func, event_func, y_position, height)
        self.components: List[Tuple[Any, Callable, Callable, Callable, int, int]] = []

        # Altura total do conteúdo
        self.content_height = 0

        # Superfície virtual para renderizar todo o conteúdo
        self.virtual_surface = None

    def add_component(
        self,
        component: Any,
        render_func: Callable,
        update_func: Callable = None,
        event_func: Callable = None
    ) -> None:
        """
        Adiciona um componente ao painel.

        Args:
            component: O componente a ser adicionado
            render_func: Função que renderiza o componente
            update_func: Função que atualiza o componente
            event_func: Função que processa eventos do componente
        """
        # Obter posição Y e altura do componente
        y_pos = getattr(component, 'y', getattr(component, 'panel_y', 0))

        # Calcular altura do componente (incluindo título se houver)
        # A maioria dos componentes renderiza título 20px acima
        component_height = self._calculate_component_height(component)

        self.components.append((
            component,
            render_func,
            update_func,
            event_func,
            y_pos,
            component_height
        ))

        # Recalcular altura total do conteúdo
        self._recalculate_content_height()

    def _calculate_component_height(self, component: Any) -> int:
        """Calcula a altura total de um componente incluindo seu título."""
        base_height = getattr(component, 'panel_height', getattr(component, 'height', 0))
        # Adicionar espaço para título (20px acima) e espaçamento após (30px)
        return 20 + base_height + 30

    def _recalculate_content_height(self) -> None:
        """Recalcula a altura total do conteúdo."""
        if not self.components:
            self.content_height = 0
            return

        # Encontrar a posição Y máxima
        max_y = 0
        for comp, _, _, _, y_pos, height in self.components:
            max_y = max(max_y, y_pos + height)

        self.content_height = max_y

        # Criar superfície virtual
        surface_height = max(self.content_height, self.height)
        self.virtual_surface = pygame.Surface((self.width, surface_height), pygame.SRCALPHA)

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Processa eventos.

        Args:
            event: Evento do pygame
        """
        # Processar scroll
        if event.type == pygame.MOUSEWHEEL:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if self._is_point_inside(mouse_x, mouse_y):
                self.scroll_offset -= event.y * self.scroll_speed
                self._clamp_scroll()
                return

        # Para eventos de mouse, ajustar coordenadas
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            mouse_x, mouse_y = event.pos
            # Ajustar Y para coordenadas virtuais
            virtual_y = mouse_y - self.y + self.scroll_offset

            # Criar evento ajustado
            adjusted_event = pygame.event.Event(
                event.type,
                {
                    'pos': (mouse_x, virtual_y),
                    'button': getattr(event, 'button', 1)
                }
            )

            # Passar para componentes
            for comp, _, _, event_func, _, _ in self.components:
                if event_func:
                    event_func(adjusted_event)
        else:
            # Outros eventos passam diretamente
            for comp, _, _, event_func, _, _ in self.components:
                if event_func:
                    event_func(event)

    def update(self, mouse_pos: Tuple[int, int]) -> None:
        """
        Atualiza componentes.

        Args:
            mouse_pos: Posição do mouse na tela
        """
        # Ajustar posição do mouse para coordenadas virtuais
        virtual_mouse_x = mouse_pos[0]
        virtual_mouse_y = mouse_pos[1] - self.y + self.scroll_offset

        for comp, _, update_func, _, _, _ in self.components:
            if update_func:
                update_func((virtual_mouse_x, virtual_mouse_y))

    def render(self, screen: pygame.Surface) -> None:
        """
        Renderiza o painel.

        Args:
            screen: Superfície da tela
        """
        if not self.virtual_surface:
            return

        # Limpar superfície virtual
        self.virtual_surface.fill((0, 0, 0, 0))

        # Renderizar todos os componentes na superfície virtual
        for comp, render_func, _, _, _, _ in self.components:
            render_func(self.virtual_surface)

        # Calcular região visível
        source_rect = pygame.Rect(
            0,
            int(self.scroll_offset),
            self.width,
            min(self.height, self.content_height - int(self.scroll_offset))
        )

        # Renderizar apenas a parte visível na tela
        screen.blit(
            self.virtual_surface,
            (self.x, self.y),
            source_rect
        )

        # Renderizar scrollbar se necessário
        if self.content_height > self.height:
            self._render_scrollbar(screen)

    def _render_scrollbar(self, screen: pygame.Surface) -> None:
        """Renderiza a scrollbar."""
        scrollbar_width = 8
        scrollbar_padding = 10

        # Posição da scrollbar (à direita do painel com espaçamento)
        sb_x = self.x + self.width + scrollbar_padding
        sb_y = self.y
        sb_height = self.height

        # Fundo da scrollbar
        pygame.draw.rect(
            screen,
            (50, 50, 50),
            (sb_x, sb_y, scrollbar_width, sb_height),
            border_radius=4
        )

        # Calcular tamanho e posição da barra
        visible_ratio = self.height / self.content_height
        bar_height = max(30, int(sb_height * visible_ratio))

        max_scroll = self.content_height - self.height
        if max_scroll > 0:
            scroll_progress = self.scroll_offset / max_scroll
        else:
            scroll_progress = 0

        bar_y = sb_y + int((sb_height - bar_height) * scroll_progress)

        # Desenhar barra
        pygame.draw.rect(
            screen,
            (150, 150, 150),
            (sb_x, bar_y, scrollbar_width, bar_height),
            border_radius=4
        )

    def _is_point_inside(self, x: int, y: int) -> bool:
        """Verifica se um ponto está dentro do painel."""
        return (
            self.x <= x < self.x + self.width and
            self.y <= y < self.y + self.height
        )

    def _clamp_scroll(self) -> None:
        """Limita o scroll aos valores válidos."""
        max_scroll = max(0, self.content_height - self.height)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))
