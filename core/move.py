"""Classe para representar um movimento no jogo de damas."""

from dataclasses import dataclass, field
from typing import List
from .position import Position


@dataclass
class Move:
    """
    Representa um movimento no jogo de damas.

    Um movimento pode ser simples (uma diagonal) ou uma sequência de capturas.

    Attributes:
        start: Posição inicial da peça
        end: Posição final da peça
        captured_positions: Lista de posições das peças capturadas neste movimento
        is_capture: Indica se é um movimento de captura
    """
    start: Position
    end: Position
    captured_positions: List[Position] = field(default_factory=list)

    @property
    def is_capture(self) -> bool:
        """
        Verifica se o movimento é uma captura.

        Returns:
            True se captura pelo menos uma peça, False caso contrário
        """
        return len(self.captured_positions) > 0

    @property
    def is_multi_capture(self) -> bool:
        """
        Verifica se o movimento captura múltiplas peças.

        Returns:
            True se captura mais de uma peça, False caso contrário
        """
        return len(self.captured_positions) > 1

    def add_captured_position(self, position: Position) -> None:
        """
        Adiciona uma posição capturada ao movimento.

        Args:
            position: Posição da peça capturada
        """
        if position not in self.captured_positions:
            self.captured_positions.append(position)

    def is_diagonal(self) -> bool:
        """
        Verifica se o movimento é diagonal.

        Returns:
            True se é diagonal, False caso contrário
        """
        row_diff = abs(self.end.row - self.start.row)
        col_diff = abs(self.end.col - self.start.col)
        return row_diff == col_diff and row_diff > 0

    def get_direction(self) -> tuple[int, int]:
        """
        Retorna a direção do movimento.

        Returns:
            Tupla (row_dir, col_dir) com -1, 0 ou 1 para cada componente
        """
        row_delta = self.end.row - self.start.row
        col_delta = self.end.col - self.start.col

        row_dir = 0 if row_delta == 0 else (1 if row_delta > 0 else -1)
        col_dir = 0 if col_delta == 0 else (1 if col_delta > 0 else -1)

        return (row_dir, col_dir)

    def get_path(self) -> List[Position]:
        """
        Retorna todas as posições no caminho do movimento (excluindo início e fim).

        Returns:
            Lista de posições no caminho
        """
        path: List[Position] = []

        row_dir, col_dir = self.get_direction()
        current_row = self.start.row + row_dir
        current_col = self.start.col + col_dir

        while current_row != self.end.row and current_col != self.end.col:
            path.append(Position(current_row, current_col))
            current_row += row_dir
            current_col += col_dir

        return path

    def __str__(self) -> str:
        """Representação em string."""
        capture_str = f" (captura {len(self.captured_positions)})" if self.is_capture else ""
        return f"{self.start} -> {self.end}{capture_str}"

    def __repr__(self) -> str:
        """Representação para debug."""
        return f"Move(start={self.start}, end={self.end}, captures={self.captured_positions})"

    def __eq__(self, other: object) -> bool:
        """Igualdade entre movimentos."""
        if not isinstance(other, Move):
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and set(self.captured_positions) == set(other.captured_positions)
        )

    def __hash__(self) -> int:
        """Hash do movimento."""
        return hash((self.start, self.end, tuple(sorted(self.captured_positions))))