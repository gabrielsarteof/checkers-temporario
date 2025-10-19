"""Classe para representar o estado do tabuleiro de damas."""

from typing import Dict, List, Optional
from copy import deepcopy
from .position import Position
from .piece import Piece
from .enums import PlayerColor, PieceType


class BoardState:
    """
    Representa o estado atual do tabuleiro de damas.

    O tabuleiro é representado por um dicionário que mapeia posições para peças.
    """

    def __init__(self):
        """Inicializa um tabuleiro vazio."""
        self.pieces: Dict[Position, Piece] = {}

    @classmethod
    def create_initial_state(cls) -> 'BoardState':
        """
        Cria o estado inicial padrão do jogo de damas.

        Peças pretas nas 3 primeiras linhas (0-2).
        Peças vermelhas nas 3 últimas linhas (5-7).
        Apenas nas casas escuras.

        Returns:
            BoardState com posição inicial do jogo
        """
        board = cls()

        # Colocar peças pretas (linhas 0-2)
        for row in range(3):
            for col in range(8):
                pos = Position(row, col)
                if pos.is_dark_square():
                    piece = Piece(PlayerColor.BLACK, PieceType.NORMAL, pos)
                    board.pieces[pos] = piece

        # Colocar peças vermelhas (linhas 5-7)
        for row in range(5, 8):
            for col in range(8):
                pos = Position(row, col)
                if pos.is_dark_square():
                    piece = Piece(PlayerColor.RED, PieceType.NORMAL, pos)
                    board.pieces[pos] = piece

        return board

    def get_piece(self, position: Position) -> Optional[Piece]:
        """
        Retorna a peça em uma posição.

        Args:
            position: Posição a verificar

        Returns:
            Peça na posição ou None se vazia
        """
        return self.pieces.get(position)

    def set_piece(self, piece: Piece) -> None:
        """
        Coloca uma peça no tabuleiro.

        Args:
            piece: Peça a colocar
        """
        self.pieces[piece.position] = piece

    def remove_piece(self, position: Position) -> Optional[Piece]:
        """
        Remove uma peça do tabuleiro.

        Args:
            position: Posição da peça a remover

        Returns:
            Peça removida ou None se posição vazia
        """
        return self.pieces.pop(position, None)

    def is_empty(self, position: Position) -> bool:
        """
        Verifica se uma posição está vazia.

        Args:
            position: Posição a verificar

        Returns:
            True se vazia, False caso contrário
        """
        return position not in self.pieces

    def get_pieces_by_color(self, color: PlayerColor) -> List[Piece]:
        """
        Retorna todas as peças de uma cor.

        Args:
            color: Cor das peças

        Returns:
            Lista de peças da cor especificada
        """
        return [piece for piece in self.pieces.values() if piece.color == color]

    def count_pieces(self, color: PlayerColor) -> int:
        """
        Conta quantas peças de uma cor existem no tabuleiro.

        Args:
            color: Cor das peças

        Returns:
            Número de peças da cor
        """
        return len(self.get_pieces_by_color(color))

    def count_kings(self, color: PlayerColor) -> int:
        """
        Conta quantas damas de uma cor existem no tabuleiro.

        Args:
            color: Cor das peças

        Returns:
            Número de damas da cor
        """
        return sum(1 for piece in self.get_pieces_by_color(color) if piece.is_king())

    def has_pieces(self, color: PlayerColor) -> bool:
        """
        Verifica se há peças de uma cor no tabuleiro.

        Args:
            color: Cor a verificar

        Returns:
            True se há pelo menos uma peça da cor, False caso contrário
        """
        return self.count_pieces(color) > 0

    def clone(self) -> 'BoardState':
        """
        Cria uma cópia profunda do estado do tabuleiro.

        Returns:
            Nova instância de BoardState com mesmo estado
        """
        new_board = BoardState()
        new_board.pieces = deepcopy(self.pieces)
        return new_board

    def get_all_pieces(self) -> List[Piece]:
        """
        Retorna todas as peças no tabuleiro.

        Returns:
            Lista com todas as peças
        """
        return list(self.pieces.values())

    def __str__(self) -> str:
        """Representação em string do tabuleiro."""
        lines = []
        for row in range(8):
            line = []
            for col in range(8):
                pos = Position(row, col)
                piece = self.get_piece(pos)
                if piece:
                    if piece.color == PlayerColor.RED:
                        line.append('r' if piece.piece_type == PieceType.NORMAL else 'R')
                    else:
                        line.append('b' if piece.piece_type == PieceType.NORMAL else 'B')
                else:
                    line.append('.' if pos.is_dark_square() else ' ')
            lines.append(' '.join(line))
        return '\n'.join(lines)

    def __repr__(self) -> str:
        """Representação para debug."""
        return f"BoardState({len(self.pieces)} pieces)"