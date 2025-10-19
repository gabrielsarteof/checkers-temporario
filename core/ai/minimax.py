"""Implementação do algoritmo Minimax com poda Alpha-Beta."""

from typing import Optional, Tuple
import math
from ..board_state import BoardState
from ..move import Move
from ..enums import PlayerColor
from ..game_rules import GameRules
from ..move_generator import MoveGenerator
from ..evaluation.base_evaluator import BaseEvaluator


class MinimaxAlphaBeta:
    """
    Implementa o algoritmo Minimax com poda Alpha-Beta.

    Usado pela IA para encontrar o melhor movimento.
    """

    def __init__(self, evaluator: BaseEvaluator, max_depth: int = 4):
        """
        Inicializa o algoritmo.

        Args:
            evaluator: Função de avaliação a usar
            max_depth: Profundidade máxima de busca
        """
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.nodes_evaluated = 0

    def find_best_move(self, board: BoardState, color: PlayerColor) -> Optional[Move]:
        """
        Encontra o melhor movimento para o jogador.

        Args:
            board: Estado atual do tabuleiro
            color: Cor do jogador

        Returns:
            Melhor movimento encontrado ou None se não há movimentos
        """
        self.nodes_evaluated = 0
        best_move = None
        best_score = -math.inf

        # Obter todos os movimentos válidos
        valid_moves = MoveGenerator.get_all_valid_moves(color, board)

        if not valid_moves:
            return None

        # Avaliar cada movimento
        for move in valid_moves:
            # Aplicar movimento
            new_board = GameRules.apply_move(board, move)

            # Avaliar posição resultante
            score = self._minimax(
                board=new_board,
                depth=self.max_depth - 1,
                alpha=-math.inf,
                beta=math.inf,
                maximizing=False,  # Próxima jogada é do oponente
                color=color
            )

            # Atualizar melhor movimento
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _minimax(
        self,
        board: BoardState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        color: PlayerColor
    ) -> float:
        """
        Implementação recursiva do Minimax com poda Alpha-Beta.

        Args:
            board: Estado atual do tabuleiro
            depth: Profundidade restante de busca
            alpha: Melhor valor garantido para o maximizador
            beta: Melhor valor garantido para o minimizador
            maximizing: True se é turno do maximizador, False para minimizador
            color: Cor do jogador original (maximizador)

        Returns:
            Avaliação da posição
        """
        self.nodes_evaluated += 1

        # Determinar cor do jogador atual
        current_color = color if maximizing else color.opposite()

        # Condições de parada

        # 1. Profundidade zero - avaliar posição
        if depth == 0:
            return self.evaluator.evaluate(board, color)

        # 2. Jogo terminou
        if GameRules.is_game_over(board, current_color):
            winner = GameRules.get_winner(board, current_color)
            if winner == color:
                # Vitória para o jogador
                return 10000 + depth  # Preferir vitórias mais rápidas
            elif winner == color.opposite():
                # Derrota
                return -10000 - depth  # Evitar derrotas rápidas
            else:
                # Empate
                return 0

        # Obter movimentos válidos
        valid_moves = MoveGenerator.get_all_valid_moves(current_color, board)

        if not valid_moves:
            # Sem movimentos, jogo terminou
            return -10000 - depth if maximizing else 10000 + depth

        if maximizing:
            # Maximizar
            max_eval = -math.inf

            for move in valid_moves:
                new_board = GameRules.apply_move(board, move)
                eval_score = self._minimax(
                    board=new_board,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=False,
                    color=color
                )

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                # Poda Beta
                if beta <= alpha:
                    break

            return max_eval

        else:
            # Minimizar
            min_eval = math.inf

            for move in valid_moves:
                new_board = GameRules.apply_move(board, move)
                eval_score = self._minimax(
                    board=new_board,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=True,
                    color=color
                )

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                # Poda Alpha
                if beta <= alpha:
                    break

            return min_eval

    def get_statistics(self) -> dict:
        """
        Retorna estatísticas da última busca.

        Returns:
            Dicionário com estatísticas
        """
        return {
            'nodes_evaluated': self.nodes_evaluated,
            'max_depth': self.max_depth
        }