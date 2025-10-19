"""Avaliador simples baseado em contagem de peças."""

from .base_evaluator import BaseEvaluator
from ..board_state import BoardState
from ..enums import PlayerColor, PieceType


class PieceCountEvaluator(BaseEvaluator):
    """
    Avaliador que conta peças.

    Peças normais valem 1 ponto.
    Damas valem 2 pontos.

    A pontuação final é a diferença entre as peças do jogador
    e as peças do adversário.
    """

    # Pesos para diferentes tipos de peças
    NORMAL_PIECE_VALUE = 1.0
    KING_PIECE_VALUE = 2.0

    def evaluate(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avalia a posição contando peças.

        Args:
            board: Estado atual do tabuleiro
            color: Cor do jogador a avaliar

        Returns:
            Diferença de pontos (peças do jogador - peças do adversário)
        """
        opponent_color = color.opposite()

        # Contar peças do jogador
        player_pieces = board.get_pieces_by_color(color)
        player_score = sum(
            self.KING_PIECE_VALUE if piece.is_king() else self.NORMAL_PIECE_VALUE
            for piece in player_pieces
        )

        # Contar peças do adversário
        opponent_pieces = board.get_pieces_by_color(opponent_color)
        opponent_score = sum(
            self.KING_PIECE_VALUE if piece.is_king() else self.NORMAL_PIECE_VALUE
            for piece in opponent_pieces
        )

        # Retornar diferença
        return player_score - opponent_score