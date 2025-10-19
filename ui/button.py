"""Componente de botão genérico."""

import pygame
from typing import Callable, Optional, Tuple
from config.colors_config import ColorsConfig


class Button:
    """
    Botão clicável genérico.

    Attributes:
        rect: Retângulo do botão
        text: Texto do botão
        callback: Função a executar ao clicar
        enabled: Se o botão está habilitado
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        callback: Optional[Callable[[], None]] = None,
        font_size: int = 20
    ):
        """
        Inicializa o botão.

        Args:
            x: Posição x do botão
            y: Posição y do botão
            width: Largura do botão
            height: Altura do botão
            text: Texto do botão
            callback: Função a chamar ao clicar
            font_size: Tamanho da fonte
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.enabled = True
        self.font = pygame.font.Font(None, font_size)
        self.hovered = False

        # Cores customizáveis
        self.normal_color = ColorsConfig.BUTTON_NORMAL
        self.hover_color = ColorsConfig.BUTTON_HOVER
        self.disabled_color = ColorsConfig.BUTTON_DISABLED

    def render(self, screen: pygame.Surface) -> None:
        """
        Renderiza o botão.

        Args:
            screen: Superfície para renderização
        """
        # Determinar cor do botão
        if not self.enabled:
            bg_color = self.disabled_color
            text_color = ColorsConfig.BUTTON_TEXT_DISABLED
        elif self.hovered:
            bg_color = self.hover_color
            text_color = ColorsConfig.BUTTON_TEXT
        else:
            bg_color = self.normal_color
            text_color = ColorsConfig.BUTTON_TEXT

        # Desenhar fundo do botão
        pygame.draw.rect(screen, bg_color, self.rect)

        # Desenhar borda
        pygame.draw.rect(
            screen,
            ColorsConfig.BUTTON_BORDER,
            self.rect,
            2  # Espessura da borda
        )

        # Desenhar texto centralizado
        text_surface = self.font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Processa um evento do pygame.

        Args:
            event: Evento a processar

        Returns:
            True se o botão foi clicado, False caso contrário
        """
        if not self.enabled:
            return False

        if event.type == pygame.MOUSEMOTION:
            # Atualizar estado hover
            self.hovered = self.rect.collidepoint(event.pos)
            return False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Verificar clique
            if self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
                return True

        return False

    def set_enabled(self, enabled: bool) -> None:
        """
        Habilita ou desabilita o botão.

        Args:
            enabled: True para habilitar, False para desabilitar
        """
        self.enabled = enabled

    def is_hovered(self, mouse_pos: Tuple[int, int]) -> bool:
        """
        Verifica se o mouse está sobre o botão.

        Args:
            mouse_pos: Posição do mouse (x, y)

        Returns:
            True se o mouse está sobre o botão
        """
        return self.rect.collidepoint(mouse_pos) and self.enabled

    def update(self, mouse_pos: Tuple[int, int]) -> None:
        """
        Atualiza o estado do botão.

        Args:
            mouse_pos: Posição do mouse (x, y)
        """
        self.hovered = self.is_hovered(mouse_pos)