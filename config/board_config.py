"""Configurações do tabuleiro de damas."""

from dataclasses import dataclass


@dataclass
class BoardConfig:
    """Configurações do tabuleiro de damas 8x8."""

    # Dimensões do tabuleiro
    ROWS: int = 8
    COLS: int = 8

    # Tamanho visual
    SQUARE_SIZE: int = 70  # Tamanho de cada casa em pixels
    BOARD_SIZE: int = ROWS * SQUARE_SIZE  # Tamanho total do tabuleiro

    # Posicionamento na janela
    BOARD_X: int = 260  # Posição X do tabuleiro na janela
    BOARD_Y: int = 60   # Posição Y do tabuleiro na janela

    # Configurações das peças
    PIECE_RADIUS: int = 25  # Raio das peças
    PIECE_BORDER_WIDTH: int = 3  # Largura da borda das peças
    KING_MARK_RADIUS: int = 10  # Raio da marca de dama (coroa)

    # Borda do tabuleiro
    BORDER_WIDTH: int = 5

    @staticmethod
    def get_square_center(row: int, col: int) -> tuple[int, int]:
        """
        Calcula o centro de uma casa do tabuleiro.

        Args:
            row: Linha da casa (0-7)
            col: Coluna da casa (0-7)

        Returns:
            Tupla (x, y) com as coordenadas do centro da casa
        """
        x = BoardConfig.BOARD_X + col * BoardConfig.SQUARE_SIZE + BoardConfig.SQUARE_SIZE // 2
        y = BoardConfig.BOARD_Y + row * BoardConfig.SQUARE_SIZE + BoardConfig.SQUARE_SIZE // 2
        return (x, y)

    @staticmethod
    def get_square_from_pos(x: int, y: int) -> tuple[int, int] | None:
        """
        Converte coordenadas de pixel para posição no tabuleiro.

        Args:
            x: Coordenada x do pixel
            y: Coordenada y do pixel

        Returns:
            Tupla (row, col) se dentro do tabuleiro, None caso contrário
        """
        # Verificar se está dentro dos limites do tabuleiro
        if x < BoardConfig.BOARD_X or x >= BoardConfig.BOARD_X + BoardConfig.BOARD_SIZE:
            return None
        if y < BoardConfig.BOARD_Y or y >= BoardConfig.BOARD_Y + BoardConfig.BOARD_SIZE:
            return None

        # Calcular linha e coluna
        col = (x - BoardConfig.BOARD_X) // BoardConfig.SQUARE_SIZE
        row = (y - BoardConfig.BOARD_Y) // BoardConfig.SQUARE_SIZE

        return (row, col)