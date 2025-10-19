"""Paleta de cores da aplicação."""

from typing import Tuple

# Tipo para representar cores RGB
Color = Tuple[int, int, int]


class ColorsConfig:
    """Paleta de cores centralizada."""

    # Cores principais
    BACKGROUND: Color = (240, 240, 240)

    # Cores do tabuleiro
    BOARD_LIGHT: Color = (240, 217, 181)  # Casas claras (bege claro)
    BOARD_DARK: Color = (181, 136, 99)     # Casas escuras (marrom)
    BOARD_BORDER: Color = (101, 67, 33)    # Borda do tabuleiro (marrom escuro)

    # Cores das peças
    PIECE_RED: Color = (220, 20, 20)       # Peças vermelhas
    PIECE_RED_BORDER: Color = (150, 0, 0)  # Borda das peças vermelhas
    PIECE_BLACK: Color = (40, 40, 40)      # Peças pretas
    PIECE_BLACK_BORDER: Color = (0, 0, 0)  # Borda das peças pretas
    PIECE_KING_MARK: Color = (255, 215, 0) # Marca de dama (dourado)

    # Cores de seleção e movimento
    SELECTED_TILE: Color = (144, 238, 144)   # Casa selecionada (verde claro)
    VALID_MOVE_TILE: Color = (200, 230, 200)  # Casa de movimento válido (verde mais claro)
    VALID_MOVE: Color = (173, 216, 230)      # Movimento válido (azul claro)
    VALID_MOVE_BORDER: Color = (0, 100, 200) # Borda movimento válido
    VALID_MOVE_INDICATOR: Color = (46, 204, 113)  # Indicador de movimento válido (verde)

    # Cores dos botões
    BUTTON_NORMAL: Color = (52, 152, 219)
    BUTTON_HOVER: Color = (41, 128, 185)
    BUTTON_SELECTED: Color = (30, 100, 150)  # Botão selecionado (azul escuro)
    BUTTON_DISABLED: Color = (149, 165, 166)
    BUTTON_TEXT: Color = (255, 255, 255)
    BUTTON_TEXT_DISABLED: Color = (127, 140, 141)
    BUTTON_BORDER: Color = (40, 116, 166)

    # Cores do texto
    TEXT: Color = (33, 33, 33)
    TEXT_SECONDARY: Color = (117, 117, 117)
    TEXT_WHITE: Color = (255, 255, 255)

    # Cores dos painéis
    PANEL_BACKGROUND: Color = (245, 245, 245)
    PANEL_BORDER: Color = (150, 150, 150)

    # Cores para indicação de turno
    TURN_RED: Color = (220, 20, 20)
    TURN_BLACK: Color = (40, 40, 40)