"""Gerenciador do estado do jogo."""

from typing import List, Optional
from .board_state import BoardState
from .move import Move
from .position import Position
from .enums import PlayerColor, GameStatus, GameMode, Difficulty
from .game_rules import GameRules
from .move_generator import MoveGenerator
from .ai.ai_player import AIPlayer
import evaluators


class GameManager:
    """
    Gerencia o estado do jogo e controla o fluxo entre jogadores.

    Attributes:
        board: Estado atual do tabuleiro
        red_player: Jogador vermelho (IA ou None para humano)
        black_player: Jogador preto (IA ou None para humano)
        current_player: Cor do jogador atual
        move_history: Histórico de movimentos
        game_status: Status atual do jogo
        game_mode: Modo de jogo atual
        difficulty: Dificuldade da IA
        selected_piece: Peça atualmente selecionada pelo jogador humano
        valid_moves_for_selected: Movimentos válidos para a peça selecionada
    """

    def __init__(
        self,
        game_mode: GameMode = GameMode.HUMAN_VS_AI,
        difficulty: Difficulty = Difficulty.MEDIUM,
        red_evaluator_name: str = None,
        black_evaluator_name: str = None
    ):
        """
        Inicializa o gerenciador do jogo.

        Args:
            game_mode: Modo de jogo (HUMAN_VS_HUMAN, HUMAN_VS_AI, AI_VS_AI)
            difficulty: Dificuldade da IA
            red_evaluator_name: Nome do avaliador para IA vermelha
            black_evaluator_name: Nome do avaliador para IA preta
        """
        self.game_mode = game_mode
        self.difficulty = difficulty

        # Usar avaliadores padrão se não especificados
        self.red_evaluator_name = red_evaluator_name or evaluators.get_default_evaluator_name()
        self.black_evaluator_name = black_evaluator_name or "Meninas Superpoderosas"

        self.board = BoardState.create_initial_state()
        self.current_player = PlayerColor.RED  # Vermelho sempre começa
        self.move_history: List[Move] = []
        self.game_status = GameStatus.PLAYING

        # Jogadores (None = humano)
        self.red_player: Optional[AIPlayer] = None
        self.black_player: Optional[AIPlayer] = None

        # Controle de IA
        self.is_ai_thinking = False
        self.last_ai_move_time = 0
        self.ai_think_delay = 500  # ms de delay visual para IA

        # Seleção de peça (para jogadores humanos)
        self.selected_piece: Optional[Position] = None
        self.valid_moves_for_selected: List[Move] = []

        # FASE 10: Threefold Repetition Detection
        # Armazenar referências aos evaluators para tracking de posições
        self._red_evaluator = None
        self._black_evaluator = None

        # Inicializar jogadores conforme o modo
        self._initialize_players()

    def _initialize_players(self) -> None:
        """Inicializa os jogadores conforme o modo de jogo."""
        # Criar avaliadores para cada cor
        red_evaluator = evaluators.create_evaluator(self.red_evaluator_name)
        black_evaluator = evaluators.create_evaluator(self.black_evaluator_name)

        # FASE 10: Armazenar referências aos evaluators
        self._red_evaluator = red_evaluator
        self._black_evaluator = black_evaluator

        # Limpar histórico de posições quando criar novos evaluators
        if hasattr(red_evaluator, 'clear_position_history'):
            red_evaluator.clear_position_history()
        if hasattr(black_evaluator, 'clear_position_history'):
            black_evaluator.clear_position_history()

        if self.game_mode == GameMode.HUMAN_VS_HUMAN:
            # Ambos são humanos
            self.red_player = None
            self.black_player = None
        elif self.game_mode == GameMode.HUMAN_VS_AI:
            # Humano é RED, IA é BLACK
            self.red_player = None
            self.black_player = AIPlayer(
                color=PlayerColor.BLACK,
                evaluator=black_evaluator,
                difficulty=self.difficulty,
                name=f"IA Preta [{self.black_evaluator_name}]"
            )
        else:  # AI_VS_AI
            # Ambos são IA - podem ter avaliadores diferentes!
            self.red_player = AIPlayer(
                color=PlayerColor.RED,
                evaluator=red_evaluator,
                difficulty=self.difficulty,
                name=f"IA Vermelha [{self.red_evaluator_name}]"
            )
            self.black_player = AIPlayer(
                color=PlayerColor.BLACK,
                evaluator=black_evaluator,
                difficulty=self.difficulty,
                name=f"IA Preta [{self.black_evaluator_name}]"
            )

    def get_current_ai_player(self) -> Optional[AIPlayer]:
        """
        Retorna o jogador de IA atual.

        Returns:
            Jogador IA da cor atual ou None se é jogador humano
        """
        return self.red_player if self.current_player == PlayerColor.RED else self.black_player

    def is_human_turn(self) -> bool:
        """
        Verifica se é turno de um jogador humano.

        Returns:
            True se é turno de humano
        """
        current_ai = self.get_current_ai_player()
        return current_ai is None

    def is_ai_turn(self) -> bool:
        """
        Verifica se é turno da IA.

        Returns:
            True se é turno da IA
        """
        current_ai = self.get_current_ai_player()
        return current_ai is not None

    def process_ai_move(self, current_time: int) -> bool:
        """
        Processa o movimento da IA com delay visual.

        Args:
            current_time: Tempo atual em milissegundos

        Returns:
            True se a IA executou um movimento
        """
        # Verificar se é turno da IA
        if not self.is_ai_turn():
            return False

        # Verificar se jogo terminou
        if self.game_status != GameStatus.PLAYING:
            return False

        # Implementar delay visual
        if not self.is_ai_thinking:
            self.is_ai_thinking = True
            self.last_ai_move_time = current_time
            return False

        # Aguardar delay
        if current_time - self.last_ai_move_time < self.ai_think_delay:
            return False

        # Executar movimento da IA
        self.is_ai_thinking = False
        return self.execute_ai_move() is not None

    def execute_ai_move(self) -> Optional[Move]:
        """
        Executa o movimento da IA atual.

        Returns:
            Movimento executado ou None se o jogo terminou
        """
        # Verificar se o jogo está em andamento
        if self.game_status != GameStatus.PLAYING:
            return None

        # Obter o jogador atual
        current_ai = self.get_current_ai_player()

        if current_ai is None:
            return None

        # IA escolhe movimento
        move = current_ai.choose_move(self.board)

        if move is None:
            # Sem movimentos, jogo terminou
            self._update_game_status()
            return None

        # Aplicar movimento
        self.apply_move(move)

        return move

    def select_piece(self, position: Position) -> bool:
        """
        Seleciona uma peça para mover.

        Args:
            position: Posição da peça a selecionar

        Returns:
            True se a peça foi selecionada com sucesso
        """
        # Verificar se o jogo está em andamento
        if self.game_status != GameStatus.PLAYING:
            return False

        # Verificar se é turno humano
        if not self.is_human_turn():
            return False

        # Verificar se há uma peça na posição
        piece = self.board.get_piece(position)
        if piece is None:
            return False

        # Verificar se a peça é do jogador atual
        if piece.color != self.current_player:
            return False

        # Selecionar peça
        self.selected_piece = position

        # Obter movimentos válidos para a peça
        self.valid_moves_for_selected = self._get_moves_for_piece(position)

        return True

    def deselect_piece(self) -> None:
        """Desseleciona a peça atual."""
        self.selected_piece = None
        self.valid_moves_for_selected = []

    def make_human_move(self, destination: Position) -> bool:
        """
        Executa um movimento humano.

        Args:
            destination: Posição de destino

        Returns:
            True se o movimento foi executado com sucesso
        """
        # Verificar se há uma peça selecionada
        if self.selected_piece is None:
            return False

        # Procurar movimento válido
        move = None
        for valid_move in self.valid_moves_for_selected:
            if valid_move.end == destination:
                move = valid_move
                break

        if move is None:
            return False

        # Aplicar movimento
        success = self.apply_move(move)

        # Desselecionar peça
        self.deselect_piece()

        return success

    def _get_moves_for_piece(self, position: Position) -> List[Move]:
        """
        Obtém todos os movimentos válidos para uma peça.

        Args:
            position: Posição da peça

        Returns:
            Lista de movimentos válidos
        """
        piece = self.board.get_piece(position)
        if piece is None:
            return []

        # Gerar movimentos de captura (prioridade)
        capture_moves = MoveGenerator.generate_capture_moves(piece, self.board)
        if capture_moves:
            return capture_moves

        # Se não há capturas, gerar movimentos simples
        simple_moves = MoveGenerator.generate_simple_moves(piece, self.board)
        return simple_moves

    def apply_move(self, move: Move) -> bool:
        """
        Aplica um movimento ao jogo.

        Args:
            move: Movimento a aplicar

        Returns:
            True se o movimento foi aplicado com sucesso
        """
        # Validar movimento
        if not GameRules.is_valid_move(self.board, move, self.current_player):
            return False

        # Aplicar movimento
        self.board = GameRules.apply_move(self.board, move)

        # Adicionar ao histórico
        self.move_history.append(move)

        # ====================================================================
        # FASE 10.5: INTELLIGENT THREEFOLD REPETITION DETECTION
        # ====================================================================
        # Adicionar posição ao histórico de ambos os evaluators
        # (necessário porque cada um rastreia independentemente)
        repetition_count = 0

        if hasattr(self._red_evaluator, 'add_position_to_history'):
            repetition_count = max(repetition_count,
                                 self._red_evaluator.add_position_to_history(self.board))

        if hasattr(self._black_evaluator, 'add_position_to_history'):
            repetition_count = max(repetition_count,
                                 self._black_evaluator.add_position_to_history(self.board))

        # FASE 10.5: Verificar threefold com análise de posição ganhável
        if repetition_count >= 3:
            # Verificar se a posição é ganhável para algum jogador
            should_draw = False

            # Verificar com o evaluator do jogador que ACABOU de jogar
            # (antes de trocar current_player)
            evaluator = (self._red_evaluator if self.current_player == PlayerColor.RED
                        else self._black_evaluator)

            if hasattr(evaluator, 'should_accept_draw_by_repetition'):
                should_draw = evaluator.should_accept_draw_by_repetition(
                    self.board,
                    self.current_player
                )

            if should_draw:
                # Empate aceito - ninguém pode ganhar
                self.game_status = GameStatus.DRAW
                return True
            # else: Posição é ganhável - continuar jogando
            #       (repetition penalty será aplicado pelo evaluator)

        # Trocar jogador
        self.current_player = self.current_player.opposite()

        # Atualizar status do jogo
        self._update_game_status()

        return True

    def _update_game_status(self) -> None:
        """Atualiza o status do jogo."""
        self.game_status = GameRules.get_game_status(self.board, self.current_player)

    def reset_game(self) -> None:
        """Reinicia o jogo com estado inicial."""
        self.board = BoardState.create_initial_state()
        self.current_player = PlayerColor.RED
        self.move_history.clear()
        self.game_status = GameStatus.PLAYING
        self.is_ai_thinking = False
        self.deselect_piece()

        # FASE 10: Limpar histórico de posições
        if hasattr(self._red_evaluator, 'clear_position_history'):
            self._red_evaluator.clear_position_history()
        if hasattr(self._black_evaluator, 'clear_position_history'):
            self._black_evaluator.clear_position_history()

    def set_game_mode(self, game_mode: GameMode) -> None:
        """
        Altera o modo de jogo.

        Args:
            game_mode: Novo modo de jogo
        """
        self.game_mode = game_mode
        self._initialize_players()
        self.reset_game()

    def set_difficulty(self, difficulty: Difficulty) -> None:
        """
        Altera a dificuldade da IA.

        Args:
            difficulty: Nova dificuldade
        """
        self.difficulty = difficulty
        self._initialize_players()

    def set_red_evaluator(self, evaluator_name: str) -> None:
        """
        Altera o avaliador da IA vermelha.

        Args:
            evaluator_name: Nome do novo avaliador
        """
        self.red_evaluator_name = evaluator_name
        self._initialize_players()

    def set_black_evaluator(self, evaluator_name: str) -> None:
        """
        Altera o avaliador da IA preta.

        Args:
            evaluator_name: Nome do novo avaliador
        """
        self.black_evaluator_name = evaluator_name
        self._initialize_players()

    def update(self, current_time: int) -> None:
        """
        Atualiza o estado do jogo (chamado no game loop).

        Args:
            current_time: Tempo atual em milissegundos
        """
        # Processar movimento da IA se for turno dela
        if self.game_status == GameStatus.PLAYING and self.is_ai_turn():
            self.process_ai_move(current_time)

    def is_game_over(self) -> bool:
        """
        Verifica se o jogo terminou.

        Returns:
            True se o jogo terminou, False caso contrário
        """
        return self.game_status != GameStatus.PLAYING

    def get_winner(self) -> Optional[PlayerColor]:
        """
        Retorna o vencedor do jogo.

        Returns:
            Cor do vencedor ou None se ainda em jogo ou empate
        """
        return self.game_status.get_winner()

    def get_last_move(self) -> Optional[Move]:
        """
        Retorna o último movimento executado.

        Returns:
            Último movimento ou None se não há movimentos
        """
        return self.move_history[-1] if self.move_history else None

    def get_move_count(self) -> int:
        """
        Retorna o número de movimentos executados.

        Returns:
            Número de movimentos
        """
        return len(self.move_history)

    def __str__(self) -> str:
        """Representação em string."""
        return (
            f"GameManager(turno={self.current_player.value}, "
            f"movimentos={self.get_move_count()}, "
            f"status={self.game_status.value})"
        )

    def __repr__(self) -> str:
        """Representação para debug."""
        return str(self)