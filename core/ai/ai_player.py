"""Jogador controlado por IA."""

import random
from typing import Optional
from ..board_state import BoardState
from ..move import Move
from ..enums import PlayerColor, Difficulty
from ..evaluation.base_evaluator import BaseEvaluator
from ..move_generator import MoveGenerator
from .minimax import MinimaxAlphaBeta


class AIPlayer:
    """
    Representa um jogador controlado por IA.

    Usa Minimax com Alpha-Beta para escolher movimentos.
    """

    def __init__(
        self,
        color: PlayerColor,
        evaluator: BaseEvaluator,
        difficulty: Difficulty = Difficulty.MEDIUM,
        name: str = "IA"
    ):
        """
        Inicializa o jogador de IA.

        Args:
            color: Cor do jogador
            evaluator: Função de avaliação a usar
            difficulty: Dificuldade da IA (controla profundidade e aleatoriedade)
            name: Nome do jogador
        """
        self.color = color
        self.evaluator = evaluator
        self.difficulty = difficulty
        self.depth = difficulty.get_max_depth()
        self.random_move_probability = difficulty.get_random_move_probability()
        self.name = name
        self.minimax = MinimaxAlphaBeta(evaluator, self.depth)

    def choose_move(self, board: BoardState) -> Optional[Move]:
        """
        Escolhe o melhor movimento para o estado atual.

        Args:
            board: Estado atual do tabuleiro

        Returns:
            Melhor movimento encontrado ou None se não há movimentos
        """
        # Obter todos os movimentos válidos
        all_moves = self._get_all_valid_moves(board)

        if not all_moves:
            return None

        # Chance de fazer movimento aleatório (para dificuldades menores)
        if random.random() < self.random_move_probability:
            return random.choice(all_moves)

        # Usar minimax para escolher melhor movimento
        move = self.minimax.find_best_move(board, self.color)
        return move

    def _get_all_valid_moves(self, board: BoardState) -> list[Move]:
        """
        Obtém todos os movimentos válidos para o jogador.

        Args:
            board: Estado do tabuleiro

        Returns:
            Lista de movimentos válidos
        """
        moves = []
        pieces = board.get_pieces_by_color(self.color)

        for piece in pieces:
            # Gerar movimentos de captura
            capture_moves = MoveGenerator.generate_capture_moves(piece, board)
            moves.extend(capture_moves)

        # Se há capturas, são obrigatórias
        if moves:
            return moves

        # Senão, gerar movimentos simples
        for piece in pieces:
            simple_moves = MoveGenerator.generate_simple_moves(piece, board)
            moves.extend(simple_moves)

        return moves

    def get_last_statistics(self) -> dict:
        """
        Retorna estatísticas da última busca.

        Returns:
            Dicionário com estatísticas
        """
        return self.minimax.get_statistics()

    def __str__(self) -> str:
        """Representação em string."""
        return f"{self.name} ({self.color.value}) - {self.evaluator}"

    def __repr__(self) -> str:
        """Representação para debug."""
        return f"AIPlayer(color={self.color}, difficulty={self.difficulty}, depth={self.depth})"