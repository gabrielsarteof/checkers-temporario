"""Renderizador do tabuleiro de damas."""

import pygame
from typing import Optional, List
from config import BoardConfig, ColorsConfig
from core.board_state import BoardState
from core.position import Position
from core.move import Move


class BoardRenderer:
    """Responsável por renderizar o tabuleiro de damas."""

    def __init__(self, screen: pygame.Surface):
        """
        Inicializa o renderizador.

        Args:
            screen: Superfície pygame para renderização
        """
        self.screen = screen

    def render(
        self,
        board: BoardState,
        selected_pos: Optional[Position] = None,
        valid_moves: Optional[List[Move]] = None
    ) -> None:
        """
        Renderiza o tabuleiro.

        Args:
            board: Estado do tabuleiro a renderizar
            selected_pos: Posição selecionada (para highlight)
            valid_moves: Movimentos válidos para destacar
        """
        # Desenhar borda do tabuleiro
        self._draw_border()

        # Desenhar todas as casas
        for row in range(BoardConfig.ROWS):
            for col in range(BoardConfig.COLS):
                pos = Position(row, col)
                self._draw_square(pos, selected_pos, valid_moves or [])

    def _draw_border(self) -> None:
        """Desenha a borda do tabuleiro."""
        border_rect = pygame.Rect(
            BoardConfig.BOARD_X - BoardConfig.BORDER_WIDTH,
            BoardConfig.BOARD_Y - BoardConfig.BORDER_WIDTH,
            BoardConfig.BOARD_SIZE + 2 * BoardConfig.BORDER_WIDTH,
            BoardConfig.BOARD_SIZE + 2 * BoardConfig.BORDER_WIDTH
        )
        pygame.draw.rect(self.screen, ColorsConfig.BOARD_BORDER, border_rect, BoardConfig.BORDER_WIDTH)

    def _draw_square(
        self,
        position: Position,
        selected_pos: Optional[Position],
        valid_moves: List[Move]
    ) -> None:
        """
        Desenha uma casa do tabuleiro.

        Args:
            position: Posição da casa
            selected_pos: Posição selecionada (para highlight)
            valid_moves: Movimentos válidos para destacar
        """
        # Calcular coordenadas da casa
        x = BoardConfig.BOARD_X + position.col * BoardConfig.SQUARE_SIZE
        y = BoardConfig.BOARD_Y + position.row * BoardConfig.SQUARE_SIZE

        # Verificar se esta posição é destino válido
        is_valid_destination = any(move.end == position for move in valid_moves)

        # Determinar cor da casa
        if selected_pos and position == selected_pos:
            color = ColorsConfig.SELECTED_TILE
        elif is_valid_destination:
            color = ColorsConfig.VALID_MOVE_TILE
        elif position.is_dark_square():
            color = ColorsConfig.BOARD_DARK
        else:
            color = ColorsConfig.BOARD_LIGHT

        # Desenhar casa
        rect = pygame.Rect(x, y, BoardConfig.SQUARE_SIZE, BoardConfig.SQUARE_SIZE)
        pygame.draw.rect(self.screen, color, rect)

        # Desenhar borda sutil
        if position.is_dark_square():
            pygame.draw.rect(
                self.screen,
                ColorsConfig.BOARD_BORDER,
                rect,
                1  # Espessura da borda
            )

        # Desenhar indicador de movimento válido (círculo)
        if is_valid_destination:
            center_x, center_y = BoardConfig.get_square_center(position.row, position.col)
            pygame.draw.circle(
                self.screen,
                ColorsConfig.VALID_MOVE_INDICATOR,
                (center_x, center_y),
                12,  # Raio do círculo
                3  # Espessura da borda
            )

    def get_position_from_mouse(self, mouse_x: int, mouse_y: int) -> Optional[Position]:
        """
        Converte coordenadas do mouse em posição do tabuleiro.

        Args:
            mouse_x: Coordenada x do mouse
            mouse_y: Coordenada y do mouse

        Returns:
            Position se dentro do tabuleiro, None caso contrário
        """
        return BoardConfig.get_square_from_pos(mouse_x, mouse_y)