"""Enumerações utilizadas no jogo de damas."""

from enum import Enum, auto


class GameMode(Enum):
    """Modos de jogo disponíveis."""
    HUMAN_VS_HUMAN = "Humano vs Humano"
    HUMAN_VS_AI = "Humano vs IA"
    AI_VS_AI = "IA vs IA"

    def has_ai(self) -> bool:
        """
        Verifica se o modo inclui IA.

        Returns:
            True se há pelo menos uma IA
        """
        return self != GameMode.HUMAN_VS_HUMAN

    def is_human_turn(self, color: 'PlayerColor') -> bool:
        """
        Verifica se é turno de um jogador humano.

        Args:
            color: Cor do jogador atual

        Returns:
            True se é turno de um humano
        """
        if self == GameMode.HUMAN_VS_HUMAN:
            return True
        elif self == GameMode.HUMAN_VS_AI:
            # Humano sempre joga com RED
            return color == PlayerColor.RED
        else:  # AI_VS_AI
            return False

    def __str__(self) -> str:
        """Representação em string do modo."""
        return self.value


class Difficulty(Enum):
    """Níveis de dificuldade da IA."""
    EASY = "Fácil"
    MEDIUM = "Médio"
    HARD = "Difícil"

    def get_max_depth(self) -> int:
        """
        Retorna a profundidade máxima de busca para o Minimax.

        Returns:
            Profundidade máxima
        """
        depths = {
            Difficulty.EASY: 2,    # Busca superficial
            Difficulty.MEDIUM: 4,  # Busca moderada
            Difficulty.HARD: 6     # Busca profunda
        }
        return depths.get(self, 4)

    def get_random_move_probability(self) -> float:
        """
        Retorna a probabilidade de fazer um movimento aleatório.

        Returns:
            Probabilidade (0.0 a 1.0)
        """
        probabilities = {
            Difficulty.EASY: 0.3,    # 30% de chance de erro
            Difficulty.MEDIUM: 0.1,  # 10% de chance de erro
            Difficulty.HARD: 0.0     # Sempre joga perfeitamente
        }
        return probabilities.get(self, 0.0)

    def __str__(self) -> str:
        """Representação em string da dificuldade."""
        return self.value


class PlayerColor(Enum):
    """Cores dos jogadores."""
    RED = "RED"
    BLACK = "BLACK"

    def opposite(self) -> 'PlayerColor':
        """
        Retorna a cor oposta.

        Returns:
            Cor oposta ao jogador atual
        """
        return PlayerColor.BLACK if self == PlayerColor.RED else PlayerColor.RED

    def __str__(self) -> str:
        """Representação em string da cor."""
        return self.value


class PieceType(Enum):
    """Tipos de peças."""
    NORMAL = "NORMAL"  # Peça comum
    KING = "KING"      # Dama

    def __str__(self) -> str:
        """Representação em string do tipo."""
        return self.value


class GameStatus(Enum):
    """Status do jogo."""
    PLAYING = "PLAYING"          # Jogo em andamento
    RED_WINS = "RED_WINS"        # Vermelho venceu
    BLACK_WINS = "BLACK_WINS"    # Preto venceu
    DRAW = "DRAW"                # Empate

    def __str__(self) -> str:
        """Representação em string do status."""
        return self.value

    def get_winner(self) -> PlayerColor | None:
        """
        Retorna o vencedor se houver.

        Returns:
            PlayerColor do vencedor ou None se empate/em jogo
        """
        if self == GameStatus.RED_WINS:
            return PlayerColor.RED
        elif self == GameStatus.BLACK_WINS:
            return PlayerColor.BLACK
        return None