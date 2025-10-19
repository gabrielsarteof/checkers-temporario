"""Classe base para avaliadores de posição."""

from abc import ABC, abstractmethod
from ..board_state import BoardState
from ..enums import PlayerColor


class BaseEvaluator(ABC):
    """
    Classe abstrata para funções de avaliação.

    Todas as funções de avaliação devem herdar desta classe e implementar
    o método evaluate().
    """

    @abstractmethod
    def evaluate(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avalia a posição do tabuleiro do ponto de vista de uma cor.

        Valores positivos indicam vantagem para a cor especificada.
        Valores negativos indicam desvantagem.

        Args:
            board: Estado atual do tabuleiro
            color: Cor do jogador a avaliar

        Returns:
            Pontuação da posição (quanto maior, melhor para a cor)
        """
        pass

    def __str__(self) -> str:
        """Representação em string do avaliador."""
        return self.__class__.__name__