"""Regras do jogo de damas."""

from .board_state import BoardState
from .move import Move
from .piece import Piece
from .enums import PlayerColor, GameStatus, PieceType
from .move_generator import MoveGenerator
from typing import Optional


class GameRules:
    """Responsável por validar e aplicar as regras do jogo de damas."""

    @staticmethod
    def apply_move(board: BoardState, move: Move) -> BoardState:
        """
        Aplica um movimento ao tabuleiro, retornando novo estado.

        Args:
            board: Estado atual do tabuleiro
            move: Movimento a aplicar

        Returns:
            Novo estado do tabuleiro após o movimento
        """
        # Clonar o tabuleiro
        new_board = board.clone()

        # Obter a peça que está se movendo
        piece = new_board.get_piece(move.start)
        if not piece:
            raise ValueError(f"Nenhuma peça na posição inicial {move.start}")

        # Remover peça da posição inicial
        new_board.remove_piece(move.start)

        # Remover peças capturadas
        for captured_pos in move.captured_positions:
            new_board.remove_piece(captured_pos)

        # Mover peça para nova posição
        moved_piece = piece.move_to(move.end)

        # Verificar se a peça deve ser promovida
        if moved_piece.should_be_promoted():
            moved_piece = moved_piece.promote_to_king()

        # Colocar peça na nova posição
        new_board.set_piece(moved_piece)

        return new_board

    @staticmethod
    def is_valid_move(board: BoardState, move: Move, color: PlayerColor) -> bool:
        """
        Verifica se um movimento é válido.

        Args:
            board: Estado atual do tabuleiro
            move: Movimento a verificar
            color: Cor do jogador

        Returns:
            True se o movimento é válido, False caso contrário
        """
        # Verificar se há uma peça na posição inicial
        piece = board.get_piece(move.start)
        if not piece or piece.color != color:
            return False

        # Obter todos os movimentos válidos para a cor
        valid_moves = MoveGenerator.get_all_valid_moves(color, board)

        # Verificar se o movimento está na lista de movimentos válidos
        return move in valid_moves

    @staticmethod
    def get_game_status(board: BoardState, current_player: PlayerColor) -> GameStatus:
        """
        Determina o status atual do jogo.

        Args:
            board: Estado atual do tabuleiro
            current_player: Jogador atual

        Returns:
            Status do jogo (PLAYING, RED_WINS, BLACK_WINS, DRAW)
        """
        # Verificar se o jogador atual tem peças
        if not board.has_pieces(current_player):
            # Jogador atual sem peças, adversário vence
            return GameStatus.RED_WINS if current_player == PlayerColor.BLACK else GameStatus.BLACK_WINS

        # Verificar se o jogador atual tem movimentos válidos
        valid_moves = MoveGenerator.get_all_valid_moves(current_player, board)
        if not valid_moves:
            # Jogador atual sem movimentos, adversário vence
            return GameStatus.RED_WINS if current_player == PlayerColor.BLACK else GameStatus.BLACK_WINS

        # Jogo continua
        return GameStatus.PLAYING

    @staticmethod
    def is_game_over(board: BoardState, current_player: PlayerColor) -> bool:
        """
        Verifica se o jogo terminou.

        Args:
            board: Estado atual do tabuleiro
            current_player: Jogador atual

        Returns:
            True se o jogo terminou, False caso contrário
        """
        status = GameRules.get_game_status(board, current_player)
        return status != GameStatus.PLAYING

    @staticmethod
    def get_winner(board: BoardState, current_player: PlayerColor) -> Optional[PlayerColor]:
        """
        Retorna o vencedor do jogo, se houver.

        Args:
            board: Estado atual do tabuleiro
            current_player: Jogador atual

        Returns:
            Cor do vencedor ou None se o jogo ainda está em andamento ou empatou
        """
        status = GameRules.get_game_status(board, current_player)
        return status.get_winner()

    @staticmethod
    def can_player_move(board: BoardState, color: PlayerColor) -> bool:
        """
        Verifica se um jogador tem movimentos disponíveis.

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            True se o jogador pode mover, False caso contrário
        """
        valid_moves = MoveGenerator.get_all_valid_moves(color, board)
        return len(valid_moves) > 0