"""Renderizador de peças de damas."""

import pygame
from config import BoardConfig, ColorsConfig
from core.piece import Piece
from core.board_state import BoardState
from core.enums import PlayerColor


class PieceRenderer:
    """Responsável por renderizar as peças do jogo."""

    def __init__(self, screen: pygame.Surface):
        """
        Inicializa o renderizador.

        Args:
            screen: Superfície pygame para renderização
        """
        self.screen = screen

    def render_all_pieces(self, board: BoardState) -> None:
        """
        Renderiza todas as peças do tabuleiro.

        Args:
            board: Estado do tabuleiro com as peças
        """
        for piece in board.get_all_pieces():
            self.render_piece(piece)

    def render_piece(self, piece: Piece) -> None:
        """
        Renderiza uma peça individual.

        Args:
            piece: Peça a renderizar
        """
        # Obter centro da casa
        center_x, center_y = BoardConfig.get_square_center(
            piece.position.row,
            piece.position.col
        )

        # Determinar cores
        if piece.color == PlayerColor.RED:
            piece_color = ColorsConfig.PIECE_RED
            border_color = ColorsConfig.PIECE_RED_BORDER
        else:
            piece_color = ColorsConfig.PIECE_BLACK
            border_color = ColorsConfig.PIECE_BLACK_BORDER

        # Desenhar círculo principal da peça
        pygame.draw.circle(
            self.screen,
            piece_color,
            (center_x, center_y),
            BoardConfig.PIECE_RADIUS
        )

        # Desenhar borda da peça
        pygame.draw.circle(
            self.screen,
            border_color,
            (center_x, center_y),
            BoardConfig.PIECE_RADIUS,
            BoardConfig.PIECE_BORDER_WIDTH
        )

        # Se for dama, desenhar marca
        if piece.is_king():
            self._draw_king_mark(center_x, center_y)

    def _draw_king_mark(self, center_x: int, center_y: int) -> None:
        """
        Desenha a marca de dama (coroa) na peça.

        Args:
            center_x: Coordenada x do centro da peça
            center_y: Coordenada y do centro da peça
        """
        # Desenhar círculo dourado menor no centro
        pygame.draw.circle(
            self.screen,
            ColorsConfig.PIECE_KING_MARK,
            (center_x, center_y),
            BoardConfig.KING_MARK_RADIUS
        )

        # Desenhar borda do círculo da coroa
        pygame.draw.circle(
            self.screen,
            ColorsConfig.PIECE_KING_MARK,
            (center_x, center_y),
            BoardConfig.KING_MARK_RADIUS,
            2
        )