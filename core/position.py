"""Classe para representar uma posição no tabuleiro."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    """
    Representa uma posição no tabuleiro de damas.

    A classe é imutável (frozen) para permitir uso como chave em dicionários.
    """
    row: int
    col: int

    def __post_init__(self):
        """Valida a posição após inicialização."""
        if not (0 <= self.row < 8):
            raise ValueError(f"Linha inválida: {self.row}. Deve estar entre 0 e 7.")
        if not (0 <= self.col < 8):
            raise ValueError(f"Coluna inválida: {self.col}. Deve estar entre 0 e 7.")

    def is_valid(self) -> bool:
        """
        Verifica se a posição está dentro dos limites do tabuleiro.

        Returns:
            True se a posição é válida, False caso contrário
        """
        return 0 <= self.row < 8 and 0 <= self.col < 8

    def is_dark_square(self) -> bool:
        """
        Verifica se a posição está em uma casa escura.

        No tabuleiro de damas, apenas casas escuras são jogáveis.
        Casa escura = soma de linha e coluna é ímpar.

        Returns:
            True se é casa escura, False caso contrário
        """
        return (self.row + self.col) % 2 == 1

    def move(self, row_delta: int, col_delta: int) -> 'Position | None':
        """
        Cria nova posição aplicando um deslocamento.

        Args:
            row_delta: Deslocamento na linha
            col_delta: Deslocamento na coluna

        Returns:
            Nova Position se válida, None caso contrário
        """
        new_row = self.row + row_delta
        new_col = self.col + col_delta

        if 0 <= new_row < 8 and 0 <= new_col < 8:
            return Position(new_row, new_col)
        return None

    def distance_to(self, other: 'Position') -> int:
        """
        Calcula distância Manhattan até outra posição.

        Args:
            other: Outra posição

        Returns:
            Distância Manhattan (soma das diferenças absolutas)
        """
        return abs(self.row - other.row) + abs(self.col - other.col)

    def diagonal_distance_to(self, other: 'Position') -> int:
        """
        Calcula distância diagonal até outra posição.

        Args:
            other: Outra posição

        Returns:
            Distância diagonal (máximo das diferenças absolutas)
        """
        return max(abs(self.row - other.row), abs(self.col - other.col))

    def is_adjacent_diagonal(self, other: 'Position') -> bool:
        """
        Verifica se outra posição é adjacente na diagonal.

        Args:
            other: Outra posição

        Returns:
            True se é adjacente diagonal, False caso contrário
        """
        row_diff = abs(self.row - other.row)
        col_diff = abs(self.col - other.col)
        return row_diff == 1 and col_diff == 1

    def __str__(self) -> str:
        """Representação em string."""
        return f"({self.row}, {self.col})"

    def __repr__(self) -> str:
        """Representação para debug."""
        return f"Position(row={self.row}, col={self.col})"