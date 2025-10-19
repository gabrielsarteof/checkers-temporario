"""Gerador de movimentos válidos para o jogo de damas."""

from typing import List, Set
from .board_state import BoardState
from .piece import Piece
from .position import Position
from .move import Move
from .enums import PlayerColor


class MoveGenerator:
    """Responsável por gerar todos os movimentos válidos para peças."""

    @staticmethod
    def generate_simple_moves(piece: Piece, board: BoardState) -> List[Move]:
        """
        Gera movimentos simples (sem captura) para uma peça.

        Peças normais movem uma diagonal para frente.
        Damas movem uma diagonal em qualquer direção.

        Args:
            piece: Peça para gerar movimentos
            board: Estado atual do tabuleiro

        Returns:
            Lista de movimentos simples válidos
        """
        moves: List[Move] = []
        current_pos = piece.position

        if piece.is_king():
            # Dama pode mover em todas as 4 direções diagonais
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # Peça normal move apenas para frente
            forward = piece.get_forward_direction()
            directions = [(forward, -1), (forward, 1)]

        for row_delta, col_delta in directions:
            new_pos = current_pos.move(row_delta, col_delta)

            if new_pos and board.is_empty(new_pos):
                move = Move(start=current_pos, end=new_pos)
                moves.append(move)

        return moves

    @staticmethod
    def get_all_simple_moves(color: PlayerColor, board: BoardState) -> List[Move]:
        """
        Retorna todos os movimentos simples possíveis para um jogador.

        Args:
            color: Cor do jogador
            board: Estado atual do tabuleiro

        Returns:
            Lista de todos os movimentos simples válidos
        """
        moves: List[Move] = []
        pieces = board.get_pieces_by_color(color)

        for piece in pieces:
            piece_moves = MoveGenerator.generate_simple_moves(piece, board)
            moves.extend(piece_moves)

        return moves

    @staticmethod
    def generate_capture_moves(piece: Piece, board: BoardState) -> List[Move]:
        """
        Gera movimentos de captura para uma peça (incluindo capturas múltiplas).

        Args:
            piece: Peça para gerar movimentos de captura
            board: Estado atual do tabuleiro

        Returns:
            Lista de movimentos de captura válidos
        """
        moves: List[Move] = []
        captured_set: Set[Position] = set()

        MoveGenerator._generate_captures_from_position(
            piece=piece,
            current_pos=piece.position,
            board=board,
            captured_so_far=captured_set,
            start_pos=piece.position,
            moves=moves
        )

        return moves

    @staticmethod
    def _generate_captures_from_position(
        piece: Piece,
        current_pos: Position,
        board: BoardState,
        captured_so_far: Set[Position],
        start_pos: Position,
        moves: List[Move]
    ) -> bool:
        """
        Gera capturas recursivamente a partir de uma posição (para capturas múltiplas).

        Args:
            piece: Peça que está capturando
            current_pos: Posição atual da peça
            board: Estado do tabuleiro
            captured_so_far: Conjunto de posições já capturadas neste movimento
            start_pos: Posição inicial do movimento
            moves: Lista para adicionar movimentos encontrados

        Returns:
            True se pelo menos uma captura foi encontrada
        """
        found_capture = False

        # Direções para captura
        if piece.is_king():
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # Peça normal pode capturar em qualquer direção diagonal
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for row_delta, col_delta in directions:
            # Posição da peça a capturar (adjacente)
            capture_pos = current_pos.move(row_delta, col_delta)
            if not capture_pos:
                continue

            # Verificar se há uma peça adversária nesta posição
            capture_piece = board.get_piece(capture_pos)
            if not capture_piece or capture_piece.color == piece.color:
                continue

            # Verificar se já capturamos esta peça neste movimento
            if capture_pos in captured_so_far:
                continue

            # Posição de destino (após pular a peça)
            landing_pos = capture_pos.move(row_delta, col_delta)
            if not landing_pos or not board.is_empty(landing_pos):
                continue

            # Encontramos uma captura válida!
            found_capture = True

            # Adicionar peça aos capturados
            new_captured = captured_so_far.copy()
            new_captured.add(capture_pos)

            # Tentar continuar capturando a partir da nova posição
            temp_moves: List[Move] = []
            can_continue = MoveGenerator._generate_captures_from_position(
                piece=piece,
                current_pos=landing_pos,
                board=board,
                captured_so_far=new_captured,
                start_pos=start_pos,
                moves=temp_moves
            )

            # Se não conseguiu continuar, este é o fim da sequência
            if not can_continue:
                move = Move(
                    start=start_pos,
                    end=landing_pos,
                    captured_positions=list(new_captured)
                )
                moves.append(move)
            else:
                # Se conseguiu continuar, os movimentos já foram adicionados
                moves.extend(temp_moves)

        return found_capture

    @staticmethod
    def get_all_capture_moves(color: PlayerColor, board: BoardState) -> List[Move]:
        """
        Retorna todos os movimentos de captura possíveis para um jogador.

        Args:
            color: Cor do jogador
            board: Estado atual do tabuleiro

        Returns:
            Lista de todos os movimentos de captura válidos
        """
        moves: List[Move] = []
        pieces = board.get_pieces_by_color(color)

        for piece in pieces:
            piece_moves = MoveGenerator.generate_capture_moves(piece, board)
            moves.extend(piece_moves)

        return moves

    @staticmethod
    def get_all_valid_moves(color: PlayerColor, board: BoardState) -> List[Move]:
        """
        Retorna todos os movimentos válidos para um jogador.

        Seguindo as regras do jogo de damas: se houver captura disponível,
        ela é obrigatória.

        Args:
            color: Cor do jogador
            board: Estado atual do tabuleiro

        Returns:
            Lista de todos os movimentos válidos (capturas se disponíveis,
            movimentos simples caso contrário)
        """
        # Verificar se há capturas disponíveis
        capture_moves = MoveGenerator.get_all_capture_moves(color, board)

        if capture_moves:
            # Captura é obrigatória
            return capture_moves

        # Sem capturas, retornar movimentos simples
        return MoveGenerator.get_all_simple_moves(color, board)