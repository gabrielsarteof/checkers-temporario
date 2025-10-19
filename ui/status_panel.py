"""Painel de status do jogo."""

import pygame
from config import ColorsConfig, UIElementConfig
from core.enums import PlayerColor, GameStatus
from core.game_manager import GameManager


class StatusPanel:
    """
    Painel que exibe informações de status do jogo.

    Exibe:
    - Turno atual
    - Contagem de peças
    - Status do jogo (vencedor ou empate)
    """

    def __init__(self, x: int, y: int, width: int, game_manager: GameManager):
        """
        Inicializa o painel de status.

        Args:
            x: Posição X do painel
            y: Posição Y do painel
            width: Largura do painel
            game_manager: Referência ao gerenciador do jogo
        """
        self.panel_x = x
        self.panel_y = y
        self.panel_width = width
        self.panel_height = 160
        self.game_manager = game_manager

        # Fontes
        self.title_font = pygame.font.Font(None, UIElementConfig.PANEL_TITLE_FONT_SIZE)
        self.text_font = pygame.font.Font(None, UIElementConfig.PANEL_TEXT_FONT_SIZE)

    def render(self, surface: pygame.Surface) -> None:
        """
        Renderiza o painel de status.

        Args:
            surface: Superfície para renderização
        """
        # Renderizar título
        title_surface = self.title_font.render("STATUS", True, ColorsConfig.TEXT)
        title_rect = title_surface.get_rect(
            topleft=(self.panel_x, self.panel_y - 20)
        )
        surface.blit(title_surface, title_rect)

        # Desenhar fundo do painel
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        pygame.draw.rect(surface, ColorsConfig.PANEL_BACKGROUND, panel_rect)
        pygame.draw.rect(surface, ColorsConfig.PANEL_BORDER, panel_rect, 2)

        # Turno atual
        y_offset = self.panel_y + 15
        player_name = "Vermelho" if self.game_manager.current_player == PlayerColor.RED else "Preto"
        player_color = (
            ColorsConfig.TURN_RED
            if self.game_manager.current_player == PlayerColor.RED
            else ColorsConfig.TURN_BLACK
        )

        turno_text = self.text_font.render(f"Turno: {player_name}", True, player_color)
        turno_rect = turno_text.get_rect(topleft=(self.panel_x + 15, y_offset))
        surface.blit(turno_text, turno_rect)

        # Peças restantes
        y_offset += 30
        red_pieces = self.game_manager.board.count_pieces(PlayerColor.RED)
        black_pieces = self.game_manager.board.count_pieces(PlayerColor.BLACK)

        pieces_text = self.text_font.render(f"Vermelhas: {red_pieces}", True, ColorsConfig.TURN_RED)
        pieces_rect = pieces_text.get_rect(topleft=(self.panel_x + 15, y_offset))
        surface.blit(pieces_text, pieces_rect)

        y_offset += 25
        pieces_text = self.text_font.render(f"Pretas: {black_pieces}", True, ColorsConfig.TURN_BLACK)
        pieces_rect = pieces_text.get_rect(topleft=(self.panel_x + 15, y_offset))
        surface.blit(pieces_text, pieces_rect)

        # Status do jogo
        y_offset += 35
        status_text = ""
        status_color = ColorsConfig.TEXT

        if self.game_manager.game_status.value == "RED_WINS":
            status_text = "VERMELHO VENCEU!"
            status_color = ColorsConfig.TURN_RED
        elif self.game_manager.game_status.value == "BLACK_WINS":
            status_text = "PRETO VENCEU!"
            status_color = ColorsConfig.TURN_BLACK
        elif self.game_manager.game_status.value == "DRAW":
            status_text = "EMPATE!"
            status_color = ColorsConfig.TEXT

        if status_text:
            status_surface = self.title_font.render(status_text, True, status_color)
            status_rect = status_surface.get_rect(
                center=(self.panel_x + self.panel_width // 2, y_offset)
            )
            surface.blit(status_surface, status_rect)

    def update(self, mouse_pos: tuple[int, int]) -> None:
        """
        Atualiza o painel (não necessário para este componente).

        Args:
            mouse_pos: Posição do mouse
        """
        pass

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Processa eventos (não necessário para este componente).

        Args:
            event: Evento a processar
        """
        pass
