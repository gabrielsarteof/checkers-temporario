"""Classe para representar uma peça de damas."""

from dataclasses import dataclass
from .enums import PlayerColor, PieceType
from .position import Position


@dataclass
class Piece:
    """
    Representa uma peça no jogo de damas.

    Attributes:
        color: Cor da peça (RED ou BLACK)
        piece_type: Tipo da peça (NORMAL ou KING)
        position: Posição atual da peça no tabuleiro
    """
    color: PlayerColor
    piece_type: PieceType
    position: Position

    def is_king(self) -> bool:
        """
        Verifica se a peça é uma dama.

        Returns:
            True se é dama, False caso contrário
        """
        return self.piece_type == PieceType.KING

    def promote_to_king(self) -> 'Piece':
        """
        Promove a peça a dama.

        Returns:
            Nova peça com tipo KING
        """
        return Piece(
            color=self.color,
            piece_type=PieceType.KING,
            position=self.position
        )

    def should_be_promoted(self) -> bool:
        """
        Verifica se a peça deve ser promovida a dama.

        Peças vermelhas são promovidas na linha 0 (topo).
        Peças pretas são promovidas na linha 7 (fundo).

        Returns:
            True se deve ser promovida, False caso contrário
        """
        if self.is_king():
            return False

        if self.color == PlayerColor.RED:
            return self.position.row == 0
        else:  # PlayerColor.BLACK
            return self.position.row == 7

    def move_to(self, new_position: Position) -> 'Piece':
        """
        Cria nova peça na posição especificada.

        Args:
            new_position: Nova posição da peça

        Returns:
            Nova peça na posição atualizada
        """
        return Piece(
            color=self.color,
            piece_type=self.piece_type,
            position=new_position
        )

    def get_forward_direction(self) -> int:
        """
        Retorna a direção de movimento para frente da peça.

        Peças vermelhas movem para cima (row decresce).
        Peças pretas movem para baixo (row cresce).

        Returns:
            -1 para vermelho (cima), +1 para preto (baixo)
        """
        return -1 if self.color == PlayerColor.RED else 1

    def __str__(self) -> str:
        """Representação em string."""
        type_str = "K" if self.is_king() else "N"
        color_str = "R" if self.color == PlayerColor.RED else "B"
        return f"{color_str}{type_str}@{self.position}"

    def __repr__(self) -> str:
        """Representação para debug."""
        return f"Piece(color={self.color}, type={self.piece_type}, pos={self.position})"