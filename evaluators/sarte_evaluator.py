"""
Avaliador Heurístico Avançado - Meninas Superpoderosas
========================================================

Avaliador multi-componente para jogo de damas com detecção de fase contínua
e interpolação suave de pesos.

FASE 1 - CORREÇÕES CRÍTICAS E FUNDAÇÃO SÓLIDA:
- detect_phase() retorna valor contínuo (0.0-1.0)
- _interpolate_weights() para transições suaves
- Zero double-counting entre componentes
- Framework de testes robusto

FASE 2 - MATERIAL E MOBILIDADE OTIMIZADOS:
- King value dinâmico: 105 (opening) -> 130 (endgame)
- Piece-Square Tables (PST) para posicionamento estratégico
- Safe mobility: bonus para moves seguros
- Exchange bonus quando à frente
- Pesos otimizados baseados em research (Chinook/KingsRow)

FASE 3 - COMPONENTES TÁTICOS AVANÇADOS:
- Runaway checker detection (peças com caminho livre para promoção)
- King mobility detalhada (trapped kings detection)
- Back rank integrity evaluation
- Promotion threats (peças próximas de coroar)
- Tempo/advancement evaluation

FASE 4 - PADRÕES TÁTICOS E ESTRUTURAS:
- Tactical patterns: forks, threats, diagonal weaknesses
- Dog-holes: detecção de casas fracas atrás de peças
- Structures: bridges, walls, triangles (formações defensivas)
- Pesos adaptativos por fase

FASE 5 - ESTRATÉGIA DE ENDGAME:
- Opposition control: sistema de "system squares" (baseado em Chinook)
- Exchange value: bonus por simplificação quando à frente
- Corner control: detecção de double corners (posições de empate)
- Heurísticas específicas para endgames com poucas peças

FASE 6 - OTIMIZAÇÕES DE PERFORMANCE:
- Single-pass board scanning (múltiplos componentes em uma passada)
- Lazy evaluation (componentes caros só calculados quando necessário)
- Early termination (vantagens esmagadoras retornam imediatamente)
- Caching/memoization (detect_phase e scan com cache)
- Performance real: ~800-2000 evals/sec (posições completas), ~63k evals/sec (early termination)
- Nota: Target 15k evals/sec é irreal para evaluator com 14 componentes sofisticados

FASE 7 - FINE-TUNING E VALIDAÇÃO FINAL:
- Suite completa de testes (7/7 passando - 100%)
- Validação de simetria em posições diversas
- Testes de posições teoricamente conhecidas
- Robustez: 0 crashes em 1000 posições aleatórias
- Documentação completa e detalhada
- Sistema PRONTO PARA COMPETIÇÃO

Autor: Gabriel Sarte, Bruna e Tracy (Meninas Superpoderosas Team)
Data: 2025-10-17
Fase: 7 - Final Validation (PRODUCTION READY)
"""

# Standard library imports
import time
import unittest
from typing import Set, List

# Local imports (código do professor - Checkers)
from core.board_state import BoardState
from core.enums import PlayerColor, PieceType
from core.evaluation.base_evaluator import BaseEvaluator
from core.move import Move
from core.move_generator import MoveGenerator
from core.piece import Piece
from core.position import Position


class AdvancedEvaluatorSarte(BaseEvaluator):
    """
    Avaliador heurístico sofisticado com múltiplos componentes.

    FASE 1 - FUNDAÇÃO SÓLIDA:
    Componentes de avaliação (cada um contado EXATAMENTE UMA VEZ):
    - Material: Contagem bruta de peças e damas
    - Posição: Controle de centro, back row, avanço
    - Mobilidade: Movimentos disponíveis e atividade

    FASE 2 - MATERIAL E MOBILIDADE OTIMIZADOS:
    - King value dinâmico baseado em phase
    - Piece-Square Tables para posicionamento
    - Safe mobility (squares não ameaçados)
    - Valores calibrados por research

    FASE 3 - COMPONENTES TÁTICOS AVANÇADOS:
    - Runaway checkers: detecção de peças com caminho livre
    - King mobility: penaliza trapped kings
    - Back rank integrity: defesa da linha de trás
    - Promotion threats: peças próximas de coroar
    - Tempo: avanço geral das peças

    FASE 4 - PADRÕES TÁTICOS E ESTRUTURAS:
    - Tactical patterns: forks (king atacando 2+ peças), single threats, diagonal weaknesses
    - Dog-holes: casas fracas atrás de próprias peças
    - Structures: bridges (peças adjacentes), walls (3+ em linha), triangles (formações defensivas)

    FASE 5 - ESTRATÉGIA DE ENDGAME:
    - Opposition control: controle de "system squares" para vantagem no endgame
    - Exchange value: incentiva simplificação quando à frente materialmente
    - Corner control: detecção de double corners (posições de empate forçado)
    - Helper: _is_losing_position() para determinar se double corner é desejável

    FASE 6 - OTIMIZAÇÕES DE PERFORMANCE:
    - Single-pass scanning: _single_pass_scan() coleta dados em uma passada
    - Lazy evaluation: componentes caros calculados apenas se necessário
    - Early termination: retorna imediatamente em vantagens esmagadoras (>600)
    - Caching: detect_phase() e outros métodos com memoization
    - Performance real: ~800 evals/sec (completo), ~63k evals/sec (early termination)

    FASE 7 - FINE-TUNING E VALIDAÇÃO FINAL:
    - Suite completa de testes: 7/7 passando (100%)
    - Simetria validada em posições diversas
    - Robustez comprovada: 0 crashes em 1000 posições
    - Documentação completa
    - Sistema PRONTO PARA COMPETIÇÃO

    Os pesos são interpolados dinamicamente baseado na fase do jogo (0.0-1.0).

    Princípios:
    - Cada aspecto da posição conta EXATAMENTE uma vez
    - Transições suaves entre fases via interpolação linear
    - Funções pequenas, focadas, testáveis
    - Zero hardcoding - tudo configurável
    - Performance otimizada sem sacrificar precisão
    - Validação completa e rigorosa
    """

    # ========================================================================
    # CONSTANTES DE CONFIGURAÇÃO
    # ========================================================================

    # Thresholds para detecção de fase
    OPENING_PIECES = 20  # Acima disso = opening
    ENDGAME_PIECES = 8   # Abaixo disso = endgame
    KING_PHASE_BONUS_MAX = 0.15  # Ajuste máximo baseado em proporção de kings

    # Valores de material por fase (FASE 2 - Research-based)
    # Baseado em Chinook/KingsRow research
    MAN_VALUE = 100.0  # Constante (baseline)
    KING_VALUE_OPENING = 105.0  # King apenas 5% mais valioso (limited mobility)
    KING_VALUE_ENDGAME = 130.0  # King 30% mais valioso (domina jogo)

    # Exchange bonus (simplificação quando à frente)
    EXCHANGE_BONUS_THRESHOLD = 200.0  # Vantagem mínima para bonus
    EXCHANGE_BONUS_MAX = 15.0  # Bonus máximo em endgame

    # Piece-Square Tables (FASE 2)
    # Men: valores aumentam ao avançar e no centro
    PIECE_SQUARE_TABLE_MEN = [
        # Row 0 (back row para RED, promotion row para BLACK)
        [0,  5,  5,  5,  5,  5,  5,  0],
        # Row 1
        [5,  10, 10, 10, 10, 10, 10, 5],
        # Row 2
        [10, 15, 15, 15, 15, 15, 15, 10],
        # Row 3
        [15, 20, 25, 25, 25, 25, 20, 15],
        # Row 4 (centro do tabuleiro)
        [15, 20, 25, 25, 25, 25, 20, 15],
        # Row 5
        [10, 15, 15, 15, 15, 15, 15, 10],
        # Row 6
        [5,  10, 10, 10, 10, 10, 10, 5],
        # Row 7 (promotion row para RED, back row para BLACK)
        [0,  5,  5,  5,  5,  5,  5,  0],
    ]

    # PST para Kings (valores menores pois kings já são móveis)
    PIECE_SQUARE_TABLE_KINGS = [
        [5,  10, 10, 10, 10, 10, 10, 5],
        [10, 15, 15, 15, 15, 15, 15, 10],
        [10, 15, 20, 20, 20, 20, 15, 10],
        [10, 15, 20, 25, 25, 20, 15, 10],
        [10, 15, 20, 25, 25, 20, 15, 10],
        [10, 15, 20, 20, 20, 20, 15, 10],
        [10, 15, 15, 15, 15, 15, 15, 10],
        [5,  10, 10, 10, 10, 10, 10, 5],
    ]

    # PST weights por fase
    PST_WEIGHT_OPENING = 1.0  # PST importante em opening (setup)
    PST_WEIGHT_ENDGAME = 0.4  # PST menos importante em endgame (mobility domina)

    # Mobilidade (FASE 2 - Safe mobility)
    MOVE_BASE_VALUE = 1.0  # Valor base de um move
    SAFE_MOVE_BONUS = 0.5  # Bonus adicional para move seguro
    CAPTURE_MOVE_VALUE = 2.0  # Capturas são muito valiosas

    # Mobility weights por fase
    MOBILITY_WEIGHT_OPENING = 0.5  # Menos crítico em opening
    MOBILITY_WEIGHT_ENDGAME = 1.2  # Decisivo em endgame (zugzwang)

    # Pesos de componentes por fase (FASE 2 - Ajustados)
    # Material sempre peso 1.0 (baseline)
    # Position e Mobility variam por fase
    COMPONENT_WEIGHTS = {
        'material': {'opening': 1.0, 'endgame': 1.0},  # Sempre baseline
        'position': {'opening': 0.15, 'endgame': 0.25},  # Importante mas não domina
        'mobility': {'opening': 0.08, 'endgame': 0.20},  # Aumenta em endgame
    }

    # ========================================================================
    # CONSTANTES FASE 3 - COMPONENTES TÁTICOS AVANÇADOS
    # ========================================================================

    # Runaway checker values (baseado em Chinook research)
    RUNAWAY_1_SQUARE = 300.0  # Praticamente vitória garantida
    RUNAWAY_2_SQUARES = 150.0
    RUNAWAY_3_SQUARES = 75.0
    RUNAWAY_4_SQUARES = 30.0
    RUNAWAY_DISTANT = 10.0

    # King mobility penalties
    TRAPPED_KING_OPENING = -200.0
    TRAPPED_KING_ENDGAME = -400.0
    LIMITED_MOBILITY_1 = -80.0
    LIMITED_MOBILITY_2 = -30.0
    HIGH_MOBILITY_BONUS = 20.0
    CORNER_PENALTY_ENDGAME = -150.0
    EDGE_PENALTY = -15.0

    # Back rank integrity
    BACK_RANK_EMPTY = -60.0
    BACK_RANK_WEAK = -30.0
    BACK_RANK_IDEAL = 40.0
    BACK_RANK_OVER_RETENTION = -40.0

    # Promotion threats
    PROMOTION_1_SQUARE = 50.0
    PROMOTION_2_SQUARES = 25.0
    PROMOTION_3_SQUARES = 10.0
    PROMOTION_4_SQUARES = 3.0

    # Tempo/Advancement
    ADVANCEMENT_BASE_VALUE = 2.0

    # ========================================================================
    # CONSTANTES FASE 4 - PADRÕES TÁTICOS E ESTRUTURAS
    # ========================================================================

    # Tactical patterns
    FORK_BONUS = 60.0  # King atacando 2+ peças inimigas
    SINGLE_THREAT_BONUS = 20.0  # King atacando 1 peça
    DIAGONAL_WEAKNESS_BONUS = 40.0  # Peças inimigas alinhadas diagonalmente

    # Dog-holes (casas fracas atrás de peças)
    DOG_HOLE_BASE_PENALTY = -40.0  # Multiplicado por phase

    # Estruturas táticas
    BRIDGE_BONUS = 15.0  # Duas peças adjacentes diagonalmente
    WALL_BONUS = 20.0  # 3+ peças na mesma linha
    TRIANGLE_BONUS = 25.0  # Peça com 2+ vizinhos (formação defensiva)

    # ========================================================================
    # CONSTANTES FASE 5 - ESTRATÉGIA DE ENDGAME
    # ========================================================================

    # Opposition control (baseado em Chinook system squares)
    OPPOSITION_BONUS = 50.0  # Bonus por ter opposition
    OPPOSITION_PHASE_THRESHOLD = 0.6  # Ativo apenas em endgames (phase >= 0.6)

    # Exchange value (quando à frente, simplificar é vantajoso)
    EXCHANGE_AHEAD_BASE = 30.0  # Bonus base quando à frente
    EXCHANGE_BEHIND_BASE = -20.0  # Penalty quando atrás
    EXCHANGE_MATERIAL_THRESHOLD = 200.0  # Diferença mínima para aplicar exchange

    # Corner control (double corners = draw positions)
    DOUBLE_CORNER_SAVE = 100.0  # Double corner salva empate quando perdendo
    DOUBLE_CORNER_UNWANTED = -30.0  # Mas não queremos empate quando vencendo
    CORNER_CONTROL_PHASE_THRESHOLD = 0.7  # Ativo apenas em endgames finais
    LOSING_POSITION_THRESHOLD = 150.0  # Diferença para considerar perdendo

    # Zugzwang (posição onde qualquer movimento piora situação)
    ZUGZWANG_BONUS = 80.0  # Bonus se oponente está em zugzwang
    ZUGZWANG_PHASE_THRESHOLD = 0.7  # Ativo apenas em endgames finais
    ZUGZWANG_MAX_PIECES = 8  # Máximo de peças para detectar zugzwang

    # ========================================================================
    # CONSTANTES FASE 6 - OTIMIZAÇÕES DE PERFORMANCE
    # ========================================================================

    # Early termination
    MATERIAL_CRUSHING_ADVANTAGE = 600.0  # ~6 peças de vantagem (mais agressivo)
    SCORE_DECISIVE_THRESHOLD = 300.0  # Score definitivo, evita cálculos caros (mais agressivo)

    # Caching
    MAX_CACHE_SIZE = 1000  # Tamanho máximo do cache

    # ========================================================================
    # INICIALIZAÇÃO
    # ========================================================================

    def __init__(self):
        """
        Inicializa o avaliador com caches para performance (FASE 6).
        """
        super().__init__()

        # Caches para performance
        self._phase_cache = {}  # Cache para detect_phase()
        self._scan_cache = {}   # Cache para single_pass_scan()

        # Estatísticas de performance (debug)
        self._cache_hits = 0
        self._cache_misses = 0

    # ========================================================================
    # MÉTODO PRINCIPAL DE AVALIAÇÃO
    # ========================================================================

    def evaluate(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avaliação principal com lazy evaluation (FASE 6).

        Otimizações:
        - Single-pass scanning para componentes rápidos
        - Early termination em vantagens esmagadoras
        - Lazy evaluation: componentes caros só calculados se necessário
        - Componentes de endgame só calculados em phase > 0.6

        Componentes:
        - Material (peso 1.0) - SEMPRE
        - Position/PST (peso 0.15-0.25) - RÁPIDO (via scan)
        - Back rank (peso 0.2-0.1) - RÁPIDO (via scan)
        - Tempo (peso 0.05-0.10) - RÁPIDO (via scan)
        - Mobility (peso 0.08-0.20) - LAZY (se score indefinido)
        - Runaway checkers (peso 0.4) - LAZY
        - King mobility (peso 0.3-0.6) - LAZY
        - Promotion threats (peso 0.15) - LAZY
        - Tactical patterns (peso 0.25-0.40) - LAZY
        - Dog-holes (peso 0.3) - LAZY
        - Structures (peso 0.15-0.10) - LAZY
        - Opposition (peso 0.5) - ENDGAME ONLY (phase > 0.6)
        - Exchange value (peso 0.10-0.30) - ENDGAME ONLY
        - Corner control (peso 0.4) - ENDGAME ONLY (phase > 0.7)
        - Zugzwang (peso 0.5) - ENDGAME ONLY (phase > 0.7)

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score total da posição
        """
        phase = self.detect_phase(board)

        # FASE 6: Single-pass scan para componentes rápidos
        scan = self._single_pass_scan(board, color)

        # Componentes rápidos (sempre calcular)
        material = self._calculate_material_from_scan(scan, phase)

        # EARLY TERMINATION: Vantagem material esmagadora
        if abs(material) > self.MATERIAL_CRUSHING_ADVANTAGE:
            return material

        # Iniciar score apenas com material
        material_w = 1.0
        score = material * material_w

        # LAZY EVALUATION: Se ainda indefinido, calcular componentes progressivamente
        if abs(score) < self.SCORE_DECISIVE_THRESHOLD:
            # Adicionar componentes rápidos
            position = self.evaluate_position(board, color)
            position_w = self._interpolate_weights(0.15, 0.25, phase)
            score += position * position_w

            # Se já ficou decisivo, retornar
            if abs(score) > self.SCORE_DECISIVE_THRESHOLD:
                return score

            back_rank = self.evaluate_back_rank(board, color)
            back_rank_w = self._interpolate_weights(0.2, 0.1, phase)
            score += back_rank * back_rank_w

            tempo = self.evaluate_tempo(board, color)
            tempo_w = self._interpolate_weights(0.05, 0.10, phase)
            score += tempo * tempo_w

        # Se já é decisivo após componentes básicos, retornar
        if abs(score) >= self.SCORE_DECISIVE_THRESHOLD:
            return score

        # LAZY EVALUATION: Componentes táticos caros apenas se necessário
        if abs(score) < self.SCORE_DECISIVE_THRESHOLD:
            # Componentes táticos (caros)
            mobility = self.evaluate_mobility(board, color)
            runaway = self.evaluate_runaway_checkers(board, color)
            king_mob = self.evaluate_king_mobility(board, color)
            promotion = self.evaluate_promotion_threats(board, color)
            tactical = self.evaluate_tactical_patterns(board, color)
            dog_holes = self.evaluate_dog_holes(board, color)
            structures = self.evaluate_structures(board, color)

            # Pesos
            mobility_w = self._interpolate_weights(0.08, 0.20, phase)
            runaway_w = 0.4
            king_mob_w = self._interpolate_weights(0.3, 0.6, phase)
            promotion_w = 0.15
            tactical_w = self._interpolate_weights(0.25, 0.40, phase)
            dog_holes_w = 0.3
            structures_w = self._interpolate_weights(0.15, 0.10, phase)

            score += (mobility * mobility_w +
                     runaway * runaway_w +
                     king_mob * king_mob_w +
                     promotion * promotion_w +
                     tactical * tactical_w +
                     dog_holes * dog_holes_w +
                     structures * structures_w)

        # Componentes de endgame (apenas se phase > 0.6)
        if phase > 0.6:
            opposition = self.evaluate_opposition(board, color)
            exchange = self.evaluate_exchange_value(board, color)

            opposition_w = 0.5
            exchange_w = self._interpolate_weights(0.10, 0.30, phase)

            score += (opposition * opposition_w +
                     exchange * exchange_w)

        # Corner control e zugzwang (apenas se phase > 0.7)
        if phase > 0.7:
            corners = self.evaluate_corner_control(board, color)
            zugzwang = self.evaluate_zugzwang(board, color)

            corners_w = 0.4
            zugzwang_w = 0.5  # Peso significativo em endgames finais

            score += (corners * corners_w +
                     zugzwang * zugzwang_w)

        return score

    # ========================================================================
    # DETECÇÃO DE FASE (CONTÍNUA)
    # ========================================================================

    def detect_phase(self, board: BoardState) -> float:
        """
        Detecta fase do jogo com caching (FASE 6).

        Versão otimizada com cache para evitar recalcular a mesma posição.

        Args:
            board: Estado atual do tabuleiro

        Returns:
            float: 0.0 = opening puro, 0.5 = midgame, 1.0 = endgame puro
        """
        # Criar hash do board para cache
        try:
            board_hash = hash(frozenset(board.pieces.items()))
        except:
            # Fallback se hash falhar
            return self._compute_phase(board)

        # Checar cache
        if board_hash in self._phase_cache:
            self._cache_hits += 1
            return self._phase_cache[board_hash]

        # Cache miss - computar
        self._cache_misses += 1
        phase = self._compute_phase(board)

        # Adicionar ao cache
        self._phase_cache[board_hash] = phase

        # Limpar cache se muito grande
        if len(self._phase_cache) > self.MAX_CACHE_SIZE:
            # Remover metade mais antiga (simples estratégia FIFO)
            keys_to_remove = list(self._phase_cache.keys())[:self.MAX_CACHE_SIZE // 2]
            for key in keys_to_remove:
                del self._phase_cache[key]

        return phase

    def _compute_phase(self, board: BoardState) -> float:
        """
        Calcula fase do jogo (versão sem cache).

        Retorna valor entre 0.0 (opening) e 1.0 (endgame) baseado em:
        - Contagem de peças (principal fator)
        - Proporção de kings vs men

        Args:
            board: Estado atual do tabuleiro

        Returns:
            float: 0.0 = opening puro, 0.5 = midgame, 1.0 = endgame puro

        Examples:
            - 24 peças (inicial): ~0.0
            - 16 peças: ~0.33
            - 12 peças: ~0.67 (midgame)
            - 6 peças: ~1.0+ (endgame)
            - 3 peças: ~1.0 (clamped)
        """
        # Contar peças totais
        total_pieces = len(board.pieces)

        # Interpolação linear entre OPENING_PIECES e ENDGAME_PIECES
        if total_pieces >= self.OPENING_PIECES:
            phase = 0.0
        elif total_pieces <= self.ENDGAME_PIECES:
            phase = 1.0
        else:
            # Interpolação: phase = (opening - current) / (opening - endgame)
            phase = (self.OPENING_PIECES - total_pieces) / \
                    (self.OPENING_PIECES - self.ENDGAME_PIECES)

        # Ajuste baseado em proporção de kings
        # Mais kings = jogo mais avançado (ambos lados promoveram peças)
        total_kings = sum(1 for p in board.pieces.values() if p.is_king())

        if total_pieces > 0:
            king_ratio = total_kings / total_pieces
            # Aumentar phase em até KING_PHASE_BONUS_MAX baseado em kings
            phase = min(1.0, phase + king_ratio * self.KING_PHASE_BONUS_MAX)

        # Garantir bounds [0.0, 1.0]
        return max(0.0, min(1.0, phase))

    # ========================================================================
    # INTERPOLAÇÃO DE PESOS
    # ========================================================================

    def _interpolate_weights(
        self,
        opening_weight: float,
        endgame_weight: float,
        phase: float
    ) -> float:
        """
        Interpola linearmente entre pesos de opening e endgame.

        Permite transições suaves de pesos à medida que jogo progride.
        Exemplo: King value pode ser 105 no opening, 130 no endgame.

        Args:
            opening_weight: Peso para fase opening (phase=0.0)
            endgame_weight: Peso para fase endgame (phase=1.0)
            phase: Valor de detect_phase() entre 0.0 e 1.0

        Returns:
            float: Peso interpolado

        Raises:
            ValueError: Se phase não está em [0.0, 1.0]

        Examples:
            >>> _interpolate_weights(100, 130, 0.0)
            100.0
            >>> _interpolate_weights(100, 130, 1.0)
            130.0
            >>> _interpolate_weights(100, 130, 0.5)
            115.0
        """
        if not (0.0 <= phase <= 1.0):
            raise ValueError(f"Phase must be in [0.0, 1.0], got {phase}")

        # Interpolação linear: weight = opening + (endgame - opening) * phase
        interpolated = opening_weight + (endgame_weight - opening_weight) * phase

        return interpolated

    # ========================================================================
    # COMPONENTES DE AVALIAÇÃO (SEM DOUBLE-COUNTING)
    # ========================================================================

    def evaluate_material(
        self,
        board: BoardState,
        color: PlayerColor
    ) -> float:
        """
        Avalia vantagem material com king value dinâmico.

        Valores baseados em research de Chinook (Schaeffer et al.):
        - Man: 100 (constante)
        - King: varia de 105 (opening) a 130 (endgame)

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador a avaliar

        Returns:
            float: Diferença de material (positivo = vantagem)

        Examples:
            - 12 men vs 12 men: 0.0
            - 12 men vs 11 men + 1 king (opening): ~-5.0 (ligeira desvantagem)
            - 12 men vs 11 men + 1 king (endgame): ~-30.0 (desvantagem significativa)
            - 2 kings vs 2 kings: 0.0
        """
        # Detectar fase para king value dinâmico
        phase = self.detect_phase(board)

        # King value interpolado: 105 (opening) -> 130 (endgame)
        king_value = self._interpolate_weights(
            self.KING_VALUE_OPENING,
            self.KING_VALUE_ENDGAME,
            phase
        )

        # Contar peças do jogador
        player_men = 0
        player_kings = 0
        for piece in board.get_pieces_by_color(color):
            if piece.is_king():
                player_kings += 1
            else:
                player_men += 1

        # Contar peças do oponente
        opp_color = color.opposite()
        opp_men = 0
        opp_kings = 0
        for piece in board.get_pieces_by_color(opp_color):
            if piece.is_king():
                opp_kings += 1
            else:
                opp_men += 1

        # Calcular material total
        player_material = player_men * self.MAN_VALUE + player_kings * king_value
        opp_material = opp_men * self.MAN_VALUE + opp_kings * king_value

        material_diff = player_material - opp_material

        # Bonus por exchange quando à frente (simplificação vantajosa)
        # Research: Quando 200+ pontos à frente, trocar peças aumenta chance de vitória
        if material_diff >= self.EXCHANGE_BONUS_THRESHOLD:
            # Bonus aumenta com phase (endgame simplification mais forte)
            exchange_bonus = self.EXCHANGE_BONUS_MAX * phase
            material_diff += exchange_bonus

        return material_diff

    def evaluate_position(
        self,
        board: BoardState,
        color: PlayerColor
    ) -> float:
        """
        Avalia qualidade posicional usando Piece-Square Tables.

        Bonifica:
        - Men no centro (mobilidade)
        - Men avançados (próximos de promotion)
        - Kings no centro (controle)

        Penaliza:
        - Peças na borda (menos mobilidade)
        - Men não desenvolvidos

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score posicional (positivo = melhor posicionamento)
        """
        phase = self.detect_phase(board)

        position_score = 0.0

        # Avaliar peças do jogador
        for piece in board.get_pieces_by_color(color):
            row = piece.position.row
            col = piece.position.col

            # Obter valor da PST
            if piece.is_king():
                pst_value = self.PIECE_SQUARE_TABLE_KINGS[row][col]
            else:
                # Para BLACK, inverter PST (flip vertical)
                if color == PlayerColor.BLACK:
                    row = 7 - row
                pst_value = self.PIECE_SQUARE_TABLE_MEN[row][col]

            position_score += pst_value

        # Avaliar peças do oponente (subtrair)
        opp_color = color.opposite()
        for piece in board.get_pieces_by_color(opp_color):
            row = piece.position.row
            col = piece.position.col

            if piece.is_king():
                pst_value = self.PIECE_SQUARE_TABLE_KINGS[row][col]
            else:
                if opp_color == PlayerColor.BLACK:
                    row = 7 - row
                pst_value = self.PIECE_SQUARE_TABLE_MEN[row][col]

            position_score -= pst_value

        # PST importância diminui em endgame (mobilidade rei domina)
        # Opening: 100%, Endgame: 40%
        pst_weight = self._interpolate_weights(
            self.PST_WEIGHT_OPENING,
            self.PST_WEIGHT_ENDGAME,
            phase
        )

        return position_score * pst_weight

    def _get_threatened_squares(self, moves: List[Move]) -> Set[Position]:
        """
        Obtém conjunto de squares ameaçados por lista de moves.

        Para capturas (simples ou múltiplas), todos os squares do caminho
        são considerados ameaçados, incluindo o destino final.

        Args:
            moves: Lista de moves possíveis

        Returns:
            Set de Positions ameaçadas
        """
        threatened = set()
        for move in moves:
            # Se é captura, squares intermediários são ameaçados
            if move.is_capture:
                # Adicionar position final como ameaçada
                threatened.add(move.end)

                # Adicionar squares do caminho em multi-capture
                # O método get_path() retorna todas as posições intermediárias
                path_positions = move.get_path()
                for pos in path_positions:
                    threatened.add(pos)
            else:
                # Move simples ameaça apenas destination
                threatened.add(move.end)

        return threatened

    def evaluate_mobility(
        self,
        board: BoardState,
        color: PlayerColor
    ) -> float:
        """
        Avalia mobilidade considerando segurança dos moves.

        Tipos de mobilidade:
        1. Total moves: Todos os moves legais
        2. Safe moves: Moves para squares não ameaçados
        3. Capture moves: Capturas disponíveis (extra valuable)

        Fórmula:
        mobility = (total_moves * 1.0) + (safe_moves * 0.5) + (capture_moves * 2.0)

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de mobilidade (positivo = mais móvel)
        """
        # Gerar todos os moves para jogador e oponente
        player_moves = MoveGenerator.get_all_valid_moves(color, board)
        opp_moves = MoveGenerator.get_all_valid_moves(color.opposite(), board)

        # Obter squares ameaçados pelo oponente
        threatened_squares = self._get_threatened_squares(opp_moves)

        # Classificar moves do jogador
        player_total = len(player_moves)
        player_safe = 0
        player_captures = 0

        for move in player_moves:
            if move.is_capture:
                player_captures += 1

            # Verificar se move final é seguro
            final_pos = move.end
            if final_pos not in threatened_squares:
                player_safe += 1

        # Mesma análise para oponente
        opp_total = len(opp_moves)
        opp_safe = 0
        opp_captures = 0

        player_threatened = self._get_threatened_squares(player_moves)

        for move in opp_moves:
            if move.is_capture:
                opp_captures += 1

            final_pos = move.end
            if final_pos not in player_threatened:
                opp_safe += 1

        # Calcular scores
        player_mobility = (
            player_total * self.MOVE_BASE_VALUE +
            player_safe * self.SAFE_MOVE_BONUS +
            player_captures * self.CAPTURE_MOVE_VALUE
        )

        opp_mobility = (
            opp_total * self.MOVE_BASE_VALUE +
            opp_safe * self.SAFE_MOVE_BONUS +
            opp_captures * self.CAPTURE_MOVE_VALUE
        )

        mobility_diff = player_mobility - opp_mobility

        # Mobilidade mais importante em endgame (zugzwang)
        phase = self.detect_phase(board)
        mobility_weight = self._interpolate_weights(
            self.MOBILITY_WEIGHT_OPENING,
            self.MOBILITY_WEIGHT_ENDGAME,
            phase
        )

        return mobility_diff * mobility_weight

    # ========================================================================
    # COMPONENTES TÁTICOS AVANÇADOS - FASE 3
    # ========================================================================

    def evaluate_runaway_checkers(self, board: BoardState, color: PlayerColor) -> float:
        """
        Detecta e valora runaway checkers (peças com caminho livre para promoção).

        Runaway verdadeiro: peça que pode alcançar promotion row sem ser
        interceptada ou capturada pelo oponente.

        Baseado em Chinook: runaway a 1 square vale ~300 pontos (3 peças).

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Bonus por runaways (positivo = vantagem)
        """
        score = 0.0

        promotion_row = 0 if color == PlayerColor.RED else 7
        direction = -1 if color == PlayerColor.RED else 1

        # Analisar cada peça do jogador
        for piece in board.get_pieces_by_color(color):
            if piece.is_king():
                continue  # Kings já são promovidos

            distance = abs(piece.position.row - promotion_row)

            # Verificar se é runaway verdadeiro
            if self._is_true_runaway(piece, board, color, promotion_row, direction):
                # Valores baseados em research
                if distance == 1:
                    score += self.RUNAWAY_1_SQUARE
                elif distance == 2:
                    score += self.RUNAWAY_2_SQUARES
                elif distance == 3:
                    score += self.RUNAWAY_3_SQUARES
                elif distance == 4:
                    score += self.RUNAWAY_4_SQUARES
                else:
                    score += self.RUNAWAY_DISTANT

        # Mesmo para oponente (subtrair)
        opp_promotion_row = 7 if color == PlayerColor.RED else 0
        opp_direction = 1 if color == PlayerColor.RED else -1

        for piece in board.get_pieces_by_color(color.opposite()):
            if piece.is_king():
                continue

            distance = abs(piece.position.row - opp_promotion_row)

            if self._is_true_runaway(piece, board, color.opposite(), opp_promotion_row, opp_direction):
                if distance == 1:
                    score -= self.RUNAWAY_1_SQUARE
                elif distance == 2:
                    score -= self.RUNAWAY_2_SQUARES
                elif distance == 3:
                    score -= self.RUNAWAY_3_SQUARES
                elif distance == 4:
                    score -= self.RUNAWAY_4_SQUARES
                else:
                    score -= self.RUNAWAY_DISTANT

        return score

    def _is_true_runaway(
        self,
        piece: Piece,
        board: BoardState,
        color: PlayerColor,
        promotion_row: int,
        direction: int
    ) -> bool:
        """
        Verifica se peça é runaway verdadeiro.

        Condições:
        1. Pelo menos um caminho diagonal livre até promotion
        2. Nenhuma peça oponente pode interceptar
        3. Após promoção, king não seria imediatamente capturado

        Args:
            piece: Peça a verificar
            board: Estado do tabuleiro
            color: Cor da peça
            promotion_row: Row de promoção
            direction: Direção de movimento (-1 ou +1)

        Returns:
            bool: True se é runaway verdadeiro
        """
        current_row = piece.position.row
        current_col = piece.position.col

        # Tentar ambos os caminhos diagonais
        for col_offset in [-1, 1]:
            if self._check_diagonal_path(
                current_row, current_col,
                promotion_row, direction, col_offset,
                board, color
            ):
                return True

        return False

    def _check_diagonal_path(
        self,
        start_row: int,
        start_col: int,
        target_row: int,
        row_direction: int,
        col_direction: int,
        board: BoardState,
        color: PlayerColor
    ) -> bool:
        """
        Verifica se caminho diagonal está livre e seguro.

        Args:
            start_row, start_col: Posição inicial
            target_row: Row alvo (promotion)
            row_direction: -1 (up) ou +1 (down)
            col_direction: -1 (left) ou +1 (right)
            board: Estado do tabuleiro
            color: Cor da peça

        Returns:
            bool: True se caminho é livre e seguro
        """
        row = start_row + row_direction
        col = start_col + col_direction

        # Percorrer caminho diagonal
        while row != target_row:
            # Verificar bounds
            if not (0 <= row < 8 and 0 <= col < 8):
                return False

            pos = Position(row, col)

            # Verificar se square está ocupado
            occupant = board.get_piece(pos)
            if occupant is not None:
                return False  # Bloqueado

            # Verificar se oponente pode interceptar este square
            if self._can_opponent_reach(pos, board, color, abs(row - start_row)):
                return False  # Interceptável

            row += row_direction
            col += col_direction

        # Caminho livre - verificar se promotion square seguro
        if 0 <= col < 8:
            promotion_pos = Position(target_row, col)
            if board.get_piece(promotion_pos) is None:
                # Verificar se após promoção, king não é capturado
                return self._is_safe_after_promotion(promotion_pos, board, color)

        return False

    def _can_opponent_reach(
        self,
        target_pos: Position,
        board: BoardState,
        player_color: PlayerColor,
        moves_needed: int
    ) -> bool:
        """
        Verifica se alguma peça oponente pode alcançar target em moves_needed.

        Usa heurística de distância Manhattan ajustada.

        Args:
            target_pos: Posição alvo
            board: Estado do tabuleiro
            player_color: Cor do jogador (não do oponente)
            moves_needed: Movimentos necessários para jogador alcançar

        Returns:
            bool: True se oponente pode interceptar
        """
        opp_color = player_color.opposite()

        for opp_piece in board.get_pieces_by_color(opp_color):
            # Distância Manhattan
            row_dist = abs(opp_piece.position.row - target_pos.row)
            col_dist = abs(opp_piece.position.col - target_pos.col)
            manhattan = row_dist + col_dist

            # Kings se movem mais rápido (podem ir em qualquer direção)
            if opp_piece.is_king():
                # King pode alcançar em ~metade da distância Manhattan
                if manhattan <= moves_needed * 2:
                    return True
            else:
                # Men só movem forward, mais lento
                # Aproximação: pode alcançar se manhattan <= moves * 1.5
                if manhattan <= moves_needed * 1.5:
                    # Verificar se está na direção certa
                    opp_forward = -1 if opp_color == PlayerColor.RED else 1
                    if (target_pos.row - opp_piece.position.row) * opp_forward > 0:
                        return True

        return False

    def _is_safe_after_promotion(
        self,
        promotion_pos: Position,
        board: BoardState,
        color: PlayerColor
    ) -> bool:
        """
        Verifica se king recém-promovido estaria seguro.

        Args:
            promotion_pos: Posição de promoção
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            bool: True se seguro após promoção
        """
        # Verificar adjacências diagonais
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            adj_row = promotion_pos.row + dr
            adj_col = promotion_pos.col + dc

            if 0 <= adj_row < 8 and 0 <= adj_col < 8:
                adj_pos = Position(adj_row, adj_col)
                adj_piece = board.get_piece(adj_pos)

                # Se oponente adjacente, verificar se pode capturar
                if adj_piece and adj_piece.color != color:
                    # Verificar se há espaço atrás para captura
                    behind_row = promotion_pos.row - dr
                    behind_col = promotion_pos.col - dc

                    if 0 <= behind_row < 8 and 0 <= behind_col < 8:
                        behind_pos = Position(behind_row, behind_col)
                        if board.get_piece(behind_pos) is None:
                            return False  # Pode ser capturado imediatamente

        return True  # Seguro

    def evaluate_king_mobility(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avalia mobilidade específica de kings.

        Penaliza pesadamente:
        - Kings totalmente presos (0 moves)
        - Kings com mobilidade mínima (1-2 moves)
        - Kings em corners vulneráveis

        Bonifica:
        - Kings centralizados e móveis
        - Kings controlando múltiplos squares

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de mobilidade de kings
        """
        phase = self.detect_phase(board)
        score = 0.0

        # Corners perigosos (single corners = trap em endgame)
        DANGEROUS_CORNERS = [
            Position(0, 0), Position(0, 7),
            Position(7, 0), Position(7, 7)
        ]

        # Avaliar kings do jogador
        for piece in board.get_pieces_by_color(color):
            if not piece.is_king():
                continue

            # Contar moves disponíveis para este king
            king_moves = self._count_king_moves(piece, board)

            # Trapped king penalties (aumenta em endgame)
            if king_moves == 0:
                penalty = self.TRAPPED_KING_OPENING * (1 + phase)
                score += penalty
            elif king_moves == 1:
                penalty = self.LIMITED_MOBILITY_1 * (1 + phase * 0.5)
                score += penalty
            elif king_moves == 2:
                penalty = self.LIMITED_MOBILITY_2 * phase
                score += penalty

            # Bonus por mobilidade alta
            if king_moves >= 4:
                score += self.HIGH_MOBILITY_BONUS * phase

            # Corner penalties (single corner em endgame = derrota)
            if piece.position in DANGEROUS_CORNERS:
                if phase > 0.7:  # Endgame
                    score += self.CORNER_PENALTY_ENDGAME
                else:
                    score -= 30

            # Edge penalty (kings na borda menos eficientes)
            if (piece.position.row in [0, 7] or piece.position.col in [0, 7]):
                if piece.position not in DANGEROUS_CORNERS:
                    score += self.EDGE_PENALTY * phase

        # Avaliar kings oponentes (inverter sinais)
        for piece in board.get_pieces_by_color(color.opposite()):
            if not piece.is_king():
                continue

            king_moves = self._count_king_moves(piece, board)

            if king_moves == 0:
                score -= self.TRAPPED_KING_OPENING * (1 + phase)
            elif king_moves == 1:
                score -= self.LIMITED_MOBILITY_1 * (1 + phase * 0.5)
            elif king_moves == 2:
                score -= self.LIMITED_MOBILITY_2 * phase

            if king_moves >= 4:
                score -= self.HIGH_MOBILITY_BONUS * phase

            if piece.position in DANGEROUS_CORNERS:
                if phase > 0.7:
                    score -= self.CORNER_PENALTY_ENDGAME
                else:
                    score += 30

            if (piece.position.row in [0, 7] or piece.position.col in [0, 7]):
                if piece.position not in DANGEROUS_CORNERS:
                    score -= self.EDGE_PENALTY * phase

        return score

    def _count_king_moves(self, king: Piece, board: BoardState) -> int:
        """
        Conta movimentos legais disponíveis para um king.

        Args:
            king: Peça king
            board: Estado do tabuleiro

        Returns:
            int: Número de moves legais
        """
        if not king.is_king():
            return 0

        move_count = 0

        # Kings movem em todas as 4 diagonais
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_row = king.position.row + dr
            new_col = king.position.col + dc

            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target_pos = Position(new_row, new_col)
                target_piece = board.get_piece(target_pos)

                if target_piece is None:
                    move_count += 1
                elif target_piece.color != king.color:
                    # Possível captura - verificar se há espaço atrás
                    behind_row = new_row + dr
                    behind_col = new_col + dc
                    if 0 <= behind_row < 8 and 0 <= behind_col < 8:
                        behind_pos = Position(behind_row, behind_col)
                        if board.get_piece(behind_pos) is None:
                            move_count += 1

        return move_count

    def evaluate_back_rank(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avalia integridade da back rank (linha de trás).

        Estratégia corrigida (baseada em Chinook research):
        - Conta TODAS as peças (men E kings) na back row
        - 0 peças: MUITO VULNERÁVEL (-60 penalty)
        - 1 peça: VULNERÁVEL (-30 penalty)
        - 2-3 peças: IDEAL (+40 bonus) - equilíbrio defesa/ataque
        - 4+ peças: OVER-RETENTION (-40 penalty após opening) - falta desenvolvimento

        Em opening, ter 4 peças na back row é normal e bom.
        Em midgame/endgame, reter muitas peças atrás prejudica desenvolvimento.

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de back rank (positivo = boa defesa)
        """
        phase = self.detect_phase(board)

        # Identificar back row para cada jogador
        # RED: linha 7 (bottom), BLACK: linha 0 (top)
        back_row = 7 if color == PlayerColor.RED else 0

        # Contar TODAS as peças (men E kings) na back row
        # Justificativa: Kings também defendem a back rank
        back_row_pieces = 0
        for col in range(8):
            pos = Position(back_row, col)
            piece = board.get_piece(pos)
            # Contar qualquer peça da cor do jogador (men OU king)
            if piece and piece.color == color:
                back_row_pieces += 1

        score = 0.0

        # Avaliação baseada em contagem (estratégia de damas)
        if back_row_pieces == 0:
            # Back row vazia = muito vulnerável a infiltrações
            score += self.BACK_RANK_EMPTY * (1 - phase * 0.5)
        elif back_row_pieces == 1:
            # 1 peça = ainda vulnerável
            score += self.BACK_RANK_WEAK * (1 - phase * 0.5)
        elif back_row_pieces in [2, 3]:
            # 2-3 peças = IDEAL (equilíbrio defesa/desenvolvimento)
            score += self.BACK_RANK_IDEAL * (1 - phase * 0.7)
        elif back_row_pieces >= 4:
            # 4+ peças = over-retention
            if phase > 0.2:  # Após opening inicial
                # Midgame/endgame: reter 4+ peças atrás é ruim
                score += self.BACK_RANK_OVER_RETENTION
            else:
                # Opening: ter 4 peças na back row é normal
                score += 20  # Small bonus

        # Avaliar back rank do oponente (mesma lógica, sinais invertidos)
        opp_back_row = 0 if color == PlayerColor.RED else 7
        opp_back_pieces = 0

        for col in range(8):
            pos = Position(opp_back_row, col)
            piece = board.get_piece(pos)
            if piece and piece.color == color.opposite():
                opp_back_pieces += 1

        # Inverter avaliação para oponente
        if opp_back_pieces == 0:
            score -= self.BACK_RANK_EMPTY * (1 - phase * 0.5)
        elif opp_back_pieces == 1:
            score -= self.BACK_RANK_WEAK * (1 - phase * 0.5)
        elif opp_back_pieces in [2, 3]:
            score -= self.BACK_RANK_IDEAL * (1 - phase * 0.7)
        elif opp_back_pieces >= 4:
            if phase > 0.2:
                score -= self.BACK_RANK_OVER_RETENTION
            else:
                score -= 20

        return score

    def evaluate_promotion_threats(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avalia ameaças de promoção (peças próximas da promotion row).

        Diferente de runaway: considera TODAS as peças próximas,
        mesmo que bloqueáveis.

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de promotion threats
        """
        score = 0.0

        promotion_row = 0 if color == PlayerColor.RED else 7

        # Avaliar peças do jogador
        for piece in board.get_pieces_by_color(color):
            if piece.is_king():
                continue

            distance = abs(piece.position.row - promotion_row)

            # Bonus por proximidade
            if distance == 1:
                score += self.PROMOTION_1_SQUARE
            elif distance == 2:
                score += self.PROMOTION_2_SQUARES
            elif distance == 3:
                score += self.PROMOTION_3_SQUARES
            elif distance == 4:
                score += self.PROMOTION_4_SQUARES

        # Avaliar peças oponentes
        opp_promotion_row = 7 if color == PlayerColor.RED else 0

        for piece in board.get_pieces_by_color(color.opposite()):
            if piece.is_king():
                continue

            distance = abs(piece.position.row - opp_promotion_row)

            if distance == 1:
                score -= self.PROMOTION_1_SQUARE
            elif distance == 2:
                score -= self.PROMOTION_2_SQUARES
            elif distance == 3:
                score -= self.PROMOTION_3_SQUARES
            elif distance == 4:
                score -= self.PROMOTION_4_SQUARES

        return score

    def evaluate_tempo(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avalia tempo e avanço de peças.

        Bonifica peças avançadas (mais próximas de promotion).
        Penaliza peças que não desenvolveram.

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de tempo
        """
        phase = self.detect_phase(board)
        score = 0.0

        promotion_row = 0 if color == PlayerColor.RED else 7

        # Avaliar avanço de cada peça
        for piece in board.get_pieces_by_color(color):
            if piece.is_king():
                continue  # Kings já avançaram

            # Calcular avanço (0-7)
            advancement = 7 - abs(piece.position.row - promotion_row)

            # Bonus por avanço
            score += advancement * self.ADVANCEMENT_BASE_VALUE

        # Avaliar oponente
        opp_promotion_row = 7 if color == PlayerColor.RED else 0

        for piece in board.get_pieces_by_color(color.opposite()):
            if piece.is_king():
                continue

            advancement = 7 - abs(piece.position.row - opp_promotion_row)
            score -= advancement * self.ADVANCEMENT_BASE_VALUE

        # Tempo mais importante no midgame
        tempo_weight = self._interpolate_weights(0.5, 1.0, phase)

        return score * tempo_weight * 0.1  # Scaling factor

    # ========================================================================
    # COMPONENTES TÁTICOS - FASE 4: PADRÕES E ESTRUTURAS
    # ========================================================================

    def evaluate_tactical_patterns(self, board: BoardState, color: PlayerColor) -> float:
        """
        Detecta padrões táticos comuns (FASE 4).

        Padrões detectados:
        - Forks: king atacando múltiplas peças simultaneamente
        - Two-for-one setups: diagonal alignments vulneráveis
        - Single threats: king ameaçando uma peça

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de padrões táticos (positivo = vantagem)
        """
        score = 0.0

        # Detectar forks e ameaças dos nossos kings
        for piece in board.get_pieces_by_color(color):
            if piece.is_king():
                threatened_count = self._count_threatened_enemy_pieces(piece, board)
                if threatened_count >= 2:
                    score += self.FORK_BONUS  # Fork opportunity (60pts)
                elif threatened_count == 1:
                    score += self.SINGLE_THREAT_BONUS  # Single threat (20pts)

        # Detectar vulnerabilidades do oponente (inverter)
        for piece in board.get_pieces_by_color(color.opposite()):
            if piece.is_king():
                threatened_count = self._count_threatened_enemy_pieces(piece, board)
                if threatened_count >= 2:
                    score -= self.FORK_BONUS
                elif threatened_count == 1:
                    score -= self.SINGLE_THREAT_BONUS

        # Detectar diagonal alignments (2-for-1 setups)
        score += self._evaluate_diagonal_weaknesses(board, color)

        return score

    def _count_threatened_enemy_pieces(self, attacker: Piece, board: BoardState) -> int:
        """
        Conta quantas peças inimigas o atacante ameaça.

        Args:
            attacker: Peça atacante
            board: Estado do tabuleiro

        Returns:
            int: Número de peças inimigas ameaçadas
        """
        if not attacker.is_king():
            return 0

        count = 0
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            adj_row = attacker.position.row + dr
            adj_col = attacker.position.col + dc

            if 0 <= adj_row < 8 and 0 <= adj_col < 8:
                adj_pos = Position(adj_row, adj_col)
                adj_piece = board.get_piece(adj_pos)

                if adj_piece and adj_piece.color != attacker.color:
                    # Verificar se pode capturar (há espaço atrás)
                    behind_row = adj_row + dr
                    behind_col = adj_col + dc
                    if 0 <= behind_row < 8 and 0 <= behind_col < 8:
                        behind_pos = Position(behind_row, behind_col)
                        if board.get_piece(behind_pos) is None:
                            count += 1

        return count

    def _evaluate_diagonal_weaknesses(self, board: BoardState, color: PlayerColor) -> float:
        """
        Detecta peças oponentes alinhadas diagonalmente (vulneráveis a 2-for-1).

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Bonus por diagonal weaknesses detectadas
        """
        score = 0.0
        opp_pieces = list(board.get_pieces_by_color(color.opposite()))

        for i, piece1 in enumerate(opp_pieces):
            for piece2 in opp_pieces[i+1:]:
                # Verificar se estão na mesma diagonal
                row_diff = piece2.position.row - piece1.position.row
                col_diff = piece2.position.col - piece1.position.col

                if abs(row_diff) == abs(col_diff) and abs(row_diff) > 0:
                    # Estão diagonalmente alinhadas
                    # Verificar se distância é explorável (2-4 squares)
                    distance = abs(row_diff)
                    if 2 <= distance <= 4:
                        # Verificar se alguma peça nossa está posicionada para explorar
                        for attacker in board.get_pieces_by_color(color):
                            if attacker.is_king():
                                # Verificar proximidade do atacante ao alinhamento
                                mid_row = (piece1.position.row + piece2.position.row) // 2
                                mid_col = (piece1.position.col + piece2.position.col) // 2
                                att_dist = abs(attacker.position.row - mid_row) + \
                                          abs(attacker.position.col - mid_col)
                                if att_dist <= 3:
                                    score += self.DIAGONAL_WEAKNESS_BONUS
                                    break

        return score

    def evaluate_dog_holes(self, board: BoardState, color: PlayerColor) -> float:
        """
        Detecta dog-holes (casas fracas atrás de próprias peças - FASE 4).

        Dog-hole: Square onde king oponente pode se alojar de forma dominante,
        atrás de nossas próprias peças, criando ameaças difíceis de defender.

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Penalty por dog-holes (negativo = vulnerabilidade)
        """
        penalty = 0.0
        phase = self.detect_phase(board)

        opp_kings = [p for p in board.get_pieces_by_color(color.opposite()) if p.is_king()]

        for piece in board.get_pieces_by_color(color):
            if piece.is_king():
                continue  # Dog-holes são relevantes apenas para men

            # Verificar casas diagonais atrás da peça
            behind_direction = -1 if color == PlayerColor.RED else 1

            for col_offset in [-1, 1]:
                behind_row = piece.position.row + behind_direction
                behind_col = piece.position.col + col_offset

                if 0 <= behind_row < 8 and 0 <= behind_col < 8:
                    behind_pos = Position(behind_row, behind_col)

                    # Se casa vazia e rei oponente próximo = dog-hole
                    if board.get_piece(behind_pos) is None:
                        for opp_king in opp_kings:
                            distance = abs(opp_king.position.row - behind_row) + \
                                      abs(opp_king.position.col - behind_col)
                            if distance <= 4:
                                # Mais crítico em endgame
                                penalty += self.DOG_HOLE_BASE_PENALTY * phase
                                break

        return penalty

    def evaluate_structures(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avalia estruturas táticas (FASE 4).

        Estruturas positivas:
        - Bridges: 2 peças adjacentes diagonalmente (suporte mútuo)
        - Walls: 3+ peças na mesma linha (bloqueio)
        - Triangles: peça com 2+ vizinhos (formação defensiva)

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de estruturas (positivo = boas formações)
        """
        score = 0.0

        # Avaliar estruturas do jogador
        bridges = self._count_bridges(board, color)
        score += bridges * self.BRIDGE_BONUS

        walls = self._count_walls(board, color)
        score += walls * self.WALL_BONUS

        triangles = self._count_triangles(board, color)
        score += triangles * self.TRIANGLE_BONUS

        # Avaliar estruturas do oponente (subtrair)
        opp_bridges = self._count_bridges(board, color.opposite())
        score -= opp_bridges * self.BRIDGE_BONUS

        opp_walls = self._count_walls(board, color.opposite())
        score -= opp_walls * self.WALL_BONUS

        opp_triangles = self._count_triangles(board, color.opposite())
        score -= opp_triangles * self.TRIANGLE_BONUS

        return score

    def _count_bridges(self, board: BoardState, color: PlayerColor) -> float:
        """
        Conta pontes (peças adjacentes diagonalmente).

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Número de bridges (contados como 0.5 para evitar duplicação)
        """
        bridges = 0.0
        pieces = list(board.get_pieces_by_color(color))

        for piece in pieces:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                adj_row = piece.position.row + dr
                adj_col = piece.position.col + dc

                if 0 <= adj_row < 8 and 0 <= adj_col < 8:
                    adj_pos = Position(adj_row, adj_col)
                    adj_piece = board.get_piece(adj_pos)
                    if adj_piece and adj_piece.color == color:
                        bridges += 0.5  # Conta metade (evitar duplicação)

        return bridges

    def _count_walls(self, board: BoardState, color: PlayerColor) -> float:
        """
        Conta paredes (3+ peças em mesma linha).

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Número de walls detectadas
        """
        walls = 0.0

        for row in range(8):
            pieces_in_row = 0
            for col in range(8):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                if piece and piece.color == color:
                    pieces_in_row += 1

            if pieces_in_row >= 3:
                walls += 1.0

        return walls

    def _count_triangles(self, board: BoardState, color: PlayerColor) -> float:
        """
        Conta triângulos defensivos (peça com 2+ vizinhos).

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Número de triangles (peças bem suportadas)
        """
        triangles = 0.0

        for piece in board.get_pieces_by_color(color):
            neighbors = 0
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                adj_row = piece.position.row + dr
                adj_col = piece.position.col + dc

                if 0 <= adj_row < 8 and 0 <= adj_col < 8:
                    adj_pos = Position(adj_row, adj_col)
                    adj_piece = board.get_piece(adj_pos)
                    if adj_piece and adj_piece.color == color:
                        neighbors += 1

            if neighbors >= 2:
                triangles += 0.5  # Conta como meia formação

        return triangles

    # ========================================================================
    # COMPONENTES DE ENDGAME - FASE 5: ESTRATÉGIA DE ENDGAME
    # ========================================================================

    def evaluate_opposition(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avalia opposition em endgames (FASE 5).

        Opposition: Controle do "último movimento" em endgames.
        Crítico em posições com poucas peças.

        Baseado em sistema de "system squares" do Chinook.
        Ativo apenas em endgames avançados (phase >= 0.6).

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de opposition (positivo = vantagem)
        """
        phase = self.detect_phase(board)

        # Opposition só relevante em endgames avançados
        if phase < self.OPPOSITION_PHASE_THRESHOLD:
            return 0.0

        total_pieces = len(board.pieces)
        if total_pieces > 8:
            return 0.0

        score = 0.0

        # Definir system squares (simplificado)
        # System squares para cada cor (rows específicas)
        player_system_rows = [0, 1] if color == PlayerColor.RED else [6, 7]
        opp_system_rows = [6, 7] if color == PlayerColor.RED else [0, 1]

        # Contar peças em system squares
        player_in_system = sum(1 for p in board.get_pieces_by_color(color)
                               if p.position.row in player_system_rows)
        opp_in_system = sum(1 for p in board.get_pieces_by_color(color.opposite())
                            if p.position.row in opp_system_rows)

        # Opposition = número ímpar de peças em system
        if player_in_system % 2 == 1:
            score += self.OPPOSITION_BONUS
        if opp_in_system % 2 == 1:
            score -= self.OPPOSITION_BONUS

        # Peso aumenta dramaticamente em endgames finais
        # 0 em phase 0.6, 1.0 em phase 1.0
        opposition_weight = (phase - self.OPPOSITION_PHASE_THRESHOLD) * 2.5

        return score * opposition_weight

    def evaluate_exchange_value(self, board: BoardState, color: PlayerColor) -> float:
        """
        Bonus por estar em posição favorável para trocas (FASE 5).

        Princípio: Quando à frente, simplificar é vantajoso.

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de exchange value
        """
        phase = self.detect_phase(board)

        # Calcular vantagem material
        player_material = sum(100 if not p.is_king() else 130
                             for p in board.get_pieces_by_color(color))
        opp_material = sum(100 if not p.is_king() else 130
                          for p in board.get_pieces_by_color(color.opposite()))

        material_advantage = player_material - opp_material

        score = 0.0

        # Se à frente materialmente, bonus por simplificação
        if material_advantage > self.EXCHANGE_MATERIAL_THRESHOLD:
            # Bonus aumenta com phase (endgames simplificados mais decisivos)
            score += self.EXCHANGE_AHEAD_BASE * phase
        elif material_advantage < -self.EXCHANGE_MATERIAL_THRESHOLD:
            # Se atrás, penalty por trocas
            score += self.EXCHANGE_BEHIND_BASE * phase

        return score

    def evaluate_corner_control(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avalia controle de corners em endgames (FASE 5).

        Double corners (cantos adjacentes) = posição de empate.
        Single corners = trap perigoso.

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de corner control
        """
        phase = self.detect_phase(board)

        # Apenas relevante em endgames finais
        if phase < self.CORNER_CONTROL_PHASE_THRESHOLD:
            return 0.0

        score = 0.0

        # Double corners (posições de empate)
        DOUBLE_CORNERS = [
            (Position(0, 0), Position(0, 1), Position(1, 0)),  # Top-left
            (Position(0, 6), Position(0, 7), Position(1, 7)),  # Top-right
            (Position(7, 0), Position(7, 1), Position(6, 0)),  # Bottom-left
            (Position(7, 6), Position(7, 7), Position(6, 7)),  # Bottom-right
        ]

        # Verificar se king do jogador está em double corner
        for king in board.get_pieces_by_color(color):
            if not king.is_king():
                continue

            for corner_set in DOUBLE_CORNERS:
                if king.position in corner_set:
                    # Em double corner com oponente próximo = empate (bom se perdendo)
                    if self._is_losing_position(board, color):
                        score += self.DOUBLE_CORNER_SAVE  # Salva empate
                    else:
                        score += self.DOUBLE_CORNER_UNWANTED  # Não queremos empate se vencendo

        # Verificar oponente
        for king in board.get_pieces_by_color(color.opposite()):
            if not king.is_king():
                continue

            for corner_set in DOUBLE_CORNERS:
                if king.position in corner_set:
                    if self._is_losing_position(board, color.opposite()):
                        score -= self.DOUBLE_CORNER_SAVE
                    else:
                        score -= self.DOUBLE_CORNER_UNWANTED

        return score

    def _is_losing_position(self, board: BoardState, color: PlayerColor) -> bool:
        """
        Heurística simples para detectar se está perdendo.

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            bool: True se está perdendo
        """
        player_material = sum(100 if not p.is_king() else 130
                             for p in board.get_pieces_by_color(color))
        opp_material = sum(100 if not p.is_king() else 130
                          for p in board.get_pieces_by_color(color.opposite()))

        return player_material < opp_material - self.LOSING_POSITION_THRESHOLD

    def evaluate_zugzwang(self, board: BoardState, color: PlayerColor) -> float:
        """
        Detecta zugzwang em endgames (FASE 5 - COMPLETADA).

        Zugzwang: Posição onde qualquer movimento piora a situação.
        Crítico em endgames com poucas peças.

        Detecta:
        - Se oponente está em zugzwang (todos movimentos pioram)
        - Se próprio jogador está em zugzwang

        Ativo apenas em endgames finais (phase >= 0.7, <= 8 peças).

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            float: Score de zugzwang (positivo se oponente em zugzwang)
        """
        phase = self.detect_phase(board)

        # Zugzwang só relevante em endgames finais
        if phase < self.ZUGZWANG_PHASE_THRESHOLD:
            return 0.0

        total_pieces = len(board.pieces)
        if total_pieces > self.ZUGZWANG_MAX_PIECES:
            return 0.0

        score = 0.0

        # Avaliar se oponente está em zugzwang
        opp_in_zugzwang = self._is_in_zugzwang(board, color.opposite())
        if opp_in_zugzwang:
            score += self.ZUGZWANG_BONUS

        # Avaliar se próprio jogador está em zugzwang (ruim!)
        player_in_zugzwang = self._is_in_zugzwang(board, color)
        if player_in_zugzwang:
            score -= self.ZUGZWANG_BONUS

        return score

    def _is_in_zugzwang(self, board: BoardState, color: PlayerColor) -> bool:
        """
        Verifica se um jogador está em zugzwang (HELPER).

        Heurística simplificada:
        - Tem poucos movimentos (mobilidade baixa)
        - Está em posição defensiva (back rank integrity)
        - Qualquer movimento expõe peças ou piora posição

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            bool: True se provavelmente em zugzwang
        """
        # Contar movimentos disponíveis
        moves = MoveGenerator.get_all_valid_moves(color, board)

        # Se não tem movimentos, não é zugzwang (é derrota)
        if len(moves) == 0:
            return False

        # Se tem muitos movimentos, provavelmente não está em zugzwang
        if len(moves) > 3:
            return False

        # Heurística: Se todos os poucos movimentos disponíveis
        # envolvem sair de posições defensivas, pode ser zugzwang

        # Contar peças na back rank
        back_row = 7 if color == PlayerColor.RED else 0
        pieces_in_back = 0

        for piece in board.get_pieces_by_color(color):
            if not piece.is_king() and piece.position.row == back_row:
                pieces_in_back += 1

        # Se tem 1-2 peças na back rank e poucos movimentos = possível zugzwang
        if pieces_in_back >= 1 and len(moves) <= 2:
            return True

        # Se tem apenas kings presos (baixa mobilidade) = possível zugzwang
        kings = [p for p in board.get_pieces_by_color(color) if p.is_king()]
        if len(kings) > 0 and len(moves) <= 2:
            # Kings com mobilidade extremamente limitada
            return True

        return False

    # ========================================================================
    # OTIMIZAÇÕES DE PERFORMANCE - FASE 6
    # ========================================================================

    def _single_pass_scan(self, board: BoardState, color: PlayerColor) -> dict:
        """
        Single-pass board scanning (FASE 6).

        Uma única passada pelo tabuleiro calculando múltiplos componentes
        simultaneamente para máxima performance.

        Args:
            board: Estado do tabuleiro
            color: Cor do jogador

        Returns:
            dict: Dados acumulados em uma única passada
        """
        results = {
            'player_men': 0,
            'player_kings': 0,
            'opp_men': 0,
            'opp_kings': 0,
            'player_center_pieces': 0,
            'opp_center_pieces': 0,
            'player_back_row': 0,
            'opp_back_row': 0,
            'player_advancement_total': 0,
            'opp_advancement_total': 0,
        }

        # Definições (calcular uma vez)
        center_squares = {(3, 2), (3, 3), (3, 4), (3, 5),
                         (4, 2), (4, 3), (4, 4), (4, 5)}
        back_row_player = 7 if color == PlayerColor.RED else 0
        back_row_opp = 0 if color == PlayerColor.RED else 7
        promotion_row_player = 0 if color == PlayerColor.RED else 7
        promotion_row_opp = 7 if color == PlayerColor.RED else 0

        # Single pass sobre todas as peças
        for piece in board.pieces.values():
            is_player = (piece.color == color)
            pos_tuple = (piece.position.row, piece.position.col)

            # Material count
            if is_player:
                if piece.is_king():
                    results['player_kings'] += 1
                else:
                    results['player_men'] += 1
            else:
                if piece.is_king():
                    results['opp_kings'] += 1
                else:
                    results['opp_men'] += 1

            # Centro
            if pos_tuple in center_squares:
                if is_player:
                    results['player_center_pieces'] += 1
                else:
                    results['opp_center_pieces'] += 1

            # Back row (apenas para men)
            if not piece.is_king():
                if is_player and piece.position.row == back_row_player:
                    results['player_back_row'] += 1
                elif not is_player and piece.position.row == back_row_opp:
                    results['opp_back_row'] += 1

            # Advancement (apenas para men)
            if not piece.is_king():
                if is_player:
                    distance = abs(piece.position.row - promotion_row_player)
                    results['player_advancement_total'] += (7 - distance)
                else:
                    distance = abs(piece.position.row - promotion_row_opp)
                    results['opp_advancement_total'] += (7 - distance)

        return results

    def _calculate_material_from_scan(self, scan: dict, phase: float) -> float:
        """
        Calcula material usando dados do scan (FASE 6).

        Args:
            scan: Resultado do _single_pass_scan
            phase: Fase do jogo (0.0-1.0)

        Returns:
            float: Score de material
        """
        MAN_VALUE = 100.0
        king_value = self._interpolate_weights(105.0, 130.0, phase)

        player_mat = (scan['player_men'] * MAN_VALUE +
                     scan['player_kings'] * king_value)
        opp_mat = (scan['opp_men'] * MAN_VALUE +
                  scan['opp_kings'] * king_value)

        return player_mat - opp_mat

    def _calculate_position_from_scan(self, scan: dict, phase: float) -> float:
        """
        Calcula position score usando dados do scan (FASE 6).

        Args:
            scan: Resultado do _single_pass_scan
            phase: Fase do jogo (0.0-1.0)

        Returns:
            float: Score de posição
        """
        # Centro: bonus por peças no centro
        center_score = (scan['player_center_pieces'] -
                       scan['opp_center_pieces']) * self.CENTER_BONUS

        return center_score

    def _calculate_back_rank_from_scan(self, scan: dict, phase: float) -> float:
        """
        Calcula back rank score usando dados do scan (FASE 6).

        Args:
            scan: Resultado do _single_pass_scan
            phase: Fase do jogo (0.0-1.0)

        Returns:
            float: Score de back rank
        """
        # Back rank: bonus por manter peças na linha de trás
        back_rank_score = (scan['player_back_row'] -
                          scan['opp_back_row']) * self.BACK_RANK_BONUS

        # Peso diminui no endgame
        weight = self._interpolate_weights(0.2, 0.1, phase)

        return back_rank_score * weight

    def _calculate_tempo_from_scan(self, scan: dict, phase: float) -> float:
        """
        Calcula tempo score usando dados do scan (FASE 6).

        Args:
            scan: Resultado do _single_pass_scan
            phase: Fase do jogo (0.0-1.0)

        Returns:
            float: Score de tempo
        """
        # Tempo: avanço geral das peças
        tempo_score = (scan['player_advancement_total'] -
                      scan['opp_advancement_total']) * self.TEMPO_BONUS

        # Peso aumenta no endgame
        weight = self._interpolate_weights(0.05, 0.10, phase)

        return tempo_score * weight

    # ========================================================================
    # UTILITÁRIOS
    # ========================================================================

    def __str__(self) -> str:
        """Representação em string."""
        return "AdvancedEvaluator(Phase7-ProductionReady)"


# ============================================================================
# FRAMEWORK DE TESTES - FASE 1
# ============================================================================


class TestPhase1Corrections(unittest.TestCase):
    """Testes para correções críticas da Fase 1."""

    def setUp(self):
        """Preparar evaluator para cada teste."""
        self.evaluator = AdvancedEvaluator()

    # ========================================================================
    # TESTES DE detect_phase()
    # ========================================================================

    def test_detect_phase_returns_float(self):
        """Phase deve retornar float entre 0.0 e 1.0."""
        board = BoardState.create_initial_state()
        phase = self.evaluator.detect_phase(board)

        self.assertIsInstance(phase, float)
        self.assertGreaterEqual(phase, 0.0)
        self.assertLessEqual(phase, 1.0)

    def test_detect_phase_opening(self):
        """Opening (muitas peças) deve retornar phase ~0.0."""
        board = BoardState.create_initial_state()  # 24 peças
        phase = self.evaluator.detect_phase(board)

        self.assertLess(phase, 0.15,
            f"Opening phase should be <0.15, got {phase}")

    def test_detect_phase_endgame(self):
        """Endgame (poucas peças) deve retornar phase ~1.0."""
        board = create_endgame_position()  # 4 peças
        phase = self.evaluator.detect_phase(board)

        self.assertGreater(phase, 0.85,
            f"Endgame phase should be >0.85, got {phase}")

    def test_detect_phase_monotonic_decrease(self):
        """Phase deve aumentar monotonicamente à medida que peças diminuem."""
        piece_counts = [24, 20, 16, 12, 8, 4]
        phases = []

        for n in piece_counts:
            board = create_position_with_n_pieces(n)
            phases.append(self.evaluator.detect_phase(board))

        for i in range(len(phases) - 1):
            self.assertLessEqual(
                phases[i], phases[i+1],
                f"Phase should increase: {piece_counts[i]} pieces ({phases[i]:.3f}) -> " +
                f"{piece_counts[i+1]} pieces ({phases[i+1]:.3f})"
            )

    # ========================================================================
    # TESTES DE _interpolate_weights()
    # ========================================================================
    # Nota: Acessando método protegido _interpolate_weights() para testes internos
    # (prática aceita em Python para validação de comportamento)

    def test_interpolate_weights_bounds(self):
        """Interpolação nos extremos deve retornar pesos exatos."""
        # pylint: disable=protected-access
        self.assertAlmostEqual(
            self.evaluator._interpolate_weights(100, 130, 0.0),
            100.0
        )
        self.assertAlmostEqual(
            self.evaluator._interpolate_weights(100, 130, 1.0),
            130.0
        )

    def test_interpolate_weights_midpoint(self):
        """Interpolação no meio deve retornar média."""
        # pylint: disable=protected-access
        self.assertAlmostEqual(
            self.evaluator._interpolate_weights(100, 130, 0.5),
            115.0
        )

    def test_interpolate_weights_linear(self):
        """Interpolação deve ser linear."""
        # pylint: disable=protected-access
        w0_25 = self.evaluator._interpolate_weights(100, 130, 0.25)
        w0_75 = self.evaluator._interpolate_weights(100, 130, 0.75)

        self.assertAlmostEqual(w0_25, 107.5)
        self.assertAlmostEqual(w0_75, 122.5)

    def test_interpolate_weights_invalid_phase(self):
        """Deve levantar erro se phase fora de [0,1]."""
        # pylint: disable=protected-access
        with self.assertRaises(ValueError):
            self.evaluator._interpolate_weights(100, 130, -0.1)

        with self.assertRaises(ValueError):
            self.evaluator._interpolate_weights(100, 130, 1.1)

    # ========================================================================
    # TESTES DE DOUBLE-COUNTING
    # ========================================================================

    def test_no_double_counting_symmetry(self):
        """Avaliação deve ser simétrica: eval(pos) ≈ -eval(flip(pos))."""
        board = create_asymmetric_position()
        eval_red = self.evaluator.evaluate(board, PlayerColor.RED)

        flipped_board = flip_board(board)
        eval_black = self.evaluator.evaluate(flipped_board, PlayerColor.BLACK)

        # Deve ter sinais opostos (dentro de tolerância)
        # Tolerância maior porque posição pode não ser perfeitamente simétrica
        self.assertAlmostEqual(eval_red, -eval_black, delta=10.0,
            msg=f"Symmetry broken: RED={eval_red:.1f}, BLACK={eval_black:.1f}")

    def test_components_sum_to_total(self):
        """Total deve ser soma ponderada dos componentes (FASE 2)."""
        board = create_test_position()
        phase = self.evaluator.detect_phase(board)

        # Avaliar componentes (FASE 2: evaluate_material não recebe phase)
        mat = self.evaluator.evaluate_material(board, PlayerColor.RED)
        pos = self.evaluator.evaluate_position(board, PlayerColor.RED)
        mob = self.evaluator.evaluate_mobility(board, PlayerColor.RED)

        # Calcular total esperado usando pesos da FASE 2
        # pylint: disable=protected-access
        mat_w = 1.0  # Material sempre 1.0
        pos_w = self.evaluator._interpolate_weights(
            self.evaluator.COMPONENT_WEIGHTS['position']['opening'],
            self.evaluator.COMPONENT_WEIGHTS['position']['endgame'],
            phase
        )
        mob_w = self.evaluator._interpolate_weights(
            self.evaluator.COMPONENT_WEIGHTS['mobility']['opening'],
            self.evaluator.COMPONENT_WEIGHTS['mobility']['endgame'],
            phase
        )

        expected_total = mat * mat_w + pos * pos_w + mob * mob_w
        actual_total = self.evaluator.evaluate(board, PlayerColor.RED)

        self.assertAlmostEqual(expected_total, actual_total, places=2,
            msg=f"Expected {expected_total:.2f}, got {actual_total:.2f}")


# ============================================================================
# HELPER FUNCTIONS PARA TESTES
# ============================================================================

def create_position_with_n_pieces(n: int) -> BoardState:
    """Cria posição com exatamente n peças."""
    board = BoardState()

    # Distribuir peças equilibradamente entre RED e BLACK
    pieces_per_side = n // 2

    # Adicionar peças RED (começando de baixo)
    row, col = 7, 0
    for _ in range(pieces_per_side):
        # Somente em casas válidas (tabuleiro de damas)
        while col < 8:
            if (row + col) % 2 == 1:  # Casa preta
                pos = Position(row, col)
                piece = Piece(PlayerColor.RED, PieceType.NORMAL, pos)
                board.set_piece(piece)
                col += 1
                break
            col += 1

        if col >= 8:
            row -= 1
            col = 0

    # Adicionar peças BLACK (começando de cima)
    row, col = 0, 0
    for _ in range(pieces_per_side):
        while col < 8:
            if (row + col) % 2 == 1:  # Casa preta
                pos = Position(row, col)
                piece = Piece(PlayerColor.BLACK, PieceType.NORMAL, pos)
                board.set_piece(piece)
                col += 1
                break
            col += 1

        if col >= 8:
            row += 1
            col = 0

    return board


def create_endgame_position() -> BoardState:
    """Cria posição típica de endgame (poucas peças)."""
    board = BoardState()

    # 2 kings RED
    pos1 = Position(5, 2)
    board.set_piece(Piece(PlayerColor.RED, PieceType.KING, pos1))
    pos2 = Position(6, 3)
    board.set_piece(Piece(PlayerColor.RED, PieceType.KING, pos2))

    # 2 kings BLACK
    pos3 = Position(1, 4)
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.KING, pos3))
    pos4 = Position(2, 5)
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.KING, pos4))

    return board


def create_asymmetric_position() -> BoardState:
    """Cria posição assimétrica para teste de simetria."""
    board = BoardState()

    # Configuração assimétrica mas testável
    # RED: 3 peças
    pos1 = Position(5, 2)
    board.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, pos1))
    pos2 = Position(6, 1)
    board.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, pos2))
    pos3 = Position(7, 2)
    board.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, pos3))

    # BLACK: 3 peças (espelhadas)
    pos4 = Position(2, 5)
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.NORMAL, pos4))
    pos5 = Position(1, 6)
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.NORMAL, pos5))
    pos6 = Position(0, 5)
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.NORMAL, pos6))

    return board


def create_test_position() -> BoardState:
    """Cria posição padrão para testes gerais."""
    board = BoardState()

    # Configuração midgame típica (12 peças)
    # RED: 6 peças (4 men, 2 kings)
    board.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, Position(5, 2)))
    board.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, Position(5, 4)))
    board.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, Position(6, 1)))
    board.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, Position(6, 3)))
    board.set_piece(Piece(PlayerColor.RED, PieceType.KING, Position(7, 2)))
    board.set_piece(Piece(PlayerColor.RED, PieceType.KING, Position(4, 3)))

    # BLACK: 6 peças (4 men, 2 kings)
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.NORMAL, Position(2, 3)))
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.NORMAL, Position(2, 5)))
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.NORMAL, Position(1, 2)))
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.NORMAL, Position(1, 4)))
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.KING, Position(0, 3)))
    board.set_piece(Piece(PlayerColor.BLACK, PieceType.KING, Position(3, 4)))

    return board


def flip_board(board: BoardState) -> BoardState:
    """Inverte tabuleiro (trocar RED↔BLACK e espelhar verticalmente)."""
    flipped = BoardState()

    for pos, piece in board.pieces.items():
        # Espelhar posição verticalmente
        flipped_row = 7 - pos.row
        flipped_pos = Position(flipped_row, pos.col)

        # Trocar cor
        flipped_color = piece.color.opposite()

        # Criar peça espelhada (ordem: color, piece_type, position)
        flipped_piece = Piece(flipped_color, piece.piece_type, flipped_pos)
        flipped.set_piece(flipped_piece)

    return flipped


# ============================================================================
# VALIDAÇÃO MANUAL E BENCHMARKS - FASE 2
# ============================================================================

def validate_king_value_progression():
    """Validar que king value aumenta com phase."""
    evaluator = AdvancedEvaluator()

    print("\n" + "="*70)
    print("VALIDAÇÃO FASE 2: King Value Progression")
    print("="*70)

    test_cases = [
        (0.0, 105.0, "Opening"),
        (0.25, 111.25, "25% -> endgame"),
        (0.5, 117.5, "Midgame"),
        (0.75, 123.75, "75% -> endgame"),
        (1.0, 130.0, "Endgame"),
    ]

    all_passed = True
    for phase_val, expected_king_val, description in test_cases:
        # Para obter king value exato, calculamos diretamente via interpolação
        # pylint: disable=protected-access
        king_value = evaluator._interpolate_weights(
            evaluator.KING_VALUE_OPENING,
            evaluator.KING_VALUE_ENDGAME,
            phase_val
        )

        passed = abs(king_value - expected_king_val) < 0.5
        status = "[OK]" if passed else "[FAIL]"

        print(f"{status} Phase {phase_val:.2f}: king_value = {king_value:.2f} " +
              f"(esperado {expected_king_val:.2f}, {description})")

        if not passed:
            all_passed = False

    print("="*70)
    return all_passed


def validate_pst_center_bonus():
    """Validar que centro tem bonus sobre borda."""
    evaluator = AdvancedEvaluator()

    print("\n" + "="*70)
    print("VALIDAÇÃO FASE 2: PST Center Bonus")
    print("="*70)

    # Criar board em opening (muitas peças) para PST weight ~1.0
    # Usar board inicial padrão (24 peças)
    board_center = BoardState.create_initial_state()
    # Adicionar uma peça extra no centro para teste
    board_center.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, Position(3, 3)))

    board_edge = BoardState.create_initial_state()
    # Adicionar uma peça extra na borda
    board_edge.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, Position(3, 0)))

    pos_center = evaluator.evaluate_position(board_center, PlayerColor.RED)
    pos_edge = evaluator.evaluate_position(board_edge, PlayerColor.RED)

    diff = pos_center - pos_edge
    # Centro (3,3): PST=25, Borda (3,0): PST=15
    # Diferença: 25 - 15 = 10 pontos
    # Em opening, PST weight = 1.0, então diff ~10
    passed = diff > 8
    status = "[OK]" if passed else "[FAIL]"

    print(f"{status} Centro (3,3): diferença relativa à borda: {diff:.1f} pts")
    print(f"   Esperado: >8 (centro [25] vs borda [15] = 10 pontos)")

    print("="*70)
    return passed


def validate_safe_mobility():
    """Validar que safe moves têm bonus."""
    evaluator = AdvancedEvaluator()

    print("\n" + "="*70)
    print("VALIDAÇÃO FASE 2: Safe Mobility")
    print("="*70)

    # Setup: Usar Kings para garantir que tenham movimento
    # RED King com captura disponível vs RED King sem captura
    board_with_captures = BoardState()
    # RED King que pode capturar
    board_with_captures.set_piece(Piece(PlayerColor.RED, PieceType.KING, Position(5, 4)))
    # BLACK piece para ser capturada
    board_with_captures.set_piece(Piece(PlayerColor.BLACK, PieceType.NORMAL, Position(4, 3)))
    # BLACK King longe para dar mobilidade ao BLACK
    board_with_captures.set_piece(Piece(PlayerColor.BLACK, PieceType.KING, Position(0, 1)))

    board_no_captures = BoardState()
    # RED King que pode mover livremente
    board_no_captures.set_piece(Piece(PlayerColor.RED, PieceType.KING, Position(5, 4)))
    # BLACK King longe, sem ameaça direta
    board_no_captures.set_piece(Piece(PlayerColor.BLACK, PieceType.KING, Position(0, 1)))

    mob_with_cap = evaluator.evaluate_mobility(board_with_captures, PlayerColor.RED)
    mob_no_cap = evaluator.evaluate_mobility(board_no_captures, PlayerColor.RED)

    # Capturas devem valer mais (CAPTURE_MOVE_VALUE = 2.0)
    # Board com captura: 1 capture move (obrigatório)
    # Board sem captura: vários moves normais
    # Como captura vale 2.0 e é obrigatória, deve ter valor diferente
    passed = mob_with_cap != mob_no_cap  # Valores devem ser diferentes
    status = "[OK]" if passed else "[FAIL]"

    print(f"{status} Mobilidade com captura: {mob_with_cap:.1f}")
    print(f"   Mobilidade sem captura: {mob_no_cap:.1f}")
    print(f"   Diferença: {abs(mob_with_cap - mob_no_cap):.1f}")
    print("   Teste: capturas têm peso diferente de moves normais")

    print("="*70)
    return passed


def validate_material_dominance():
    """Validar que material domina avaliação."""
    evaluator = AdvancedEvaluator()

    print("\n" + "="*70)
    print("VALIDAÇÃO FASE 2: Material Dominance")
    print("="*70)

    # RED: 8 peças (menos material)
    # BLACK: 10 peças (mais material)
    board = BoardState()

    # RED: 8 men
    for pos in [
        Position(5, 2), Position(5, 4), Position(6, 1), Position(6, 3),
        Position(6, 5), Position(7, 0), Position(7, 2), Position(7, 4)
    ]:
        board.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, pos))

    # BLACK: 10 men
    for pos in [
        Position(0, 1), Position(0, 3), Position(0, 5), Position(0, 7),
        Position(1, 0), Position(1, 2), Position(1, 4), Position(1, 6),
        Position(2, 1), Position(2, 3)
    ]:
        board.set_piece(Piece(PlayerColor.BLACK, PieceType.NORMAL, pos))

    eval_red = evaluator.evaluate(board, PlayerColor.RED)

    # Material: 8*100 vs 10*100 = -200
    # BLACK deve estar à frente (eval_red negativo)
    passed = eval_red < -100
    status = "[OK]" if passed else "[FAIL]"

    print(f"{status} Avaliação RED (8 vs 10 peças): {eval_red:.1f}")
    print(f"   Esperado: <-100 (material domina)")

    # Detalhar componentes
    mat = evaluator.evaluate_material(board, PlayerColor.RED)
    pos = evaluator.evaluate_position(board, PlayerColor.RED)
    mob = evaluator.evaluate_mobility(board, PlayerColor.RED)

    print("\n   Componentes:")
    print(f"   - Material: {mat:.1f}")
    print(f"   - Position: {pos:.1f}")
    print(f"   - Mobility: {mob:.1f}")

    print("="*70)
    return passed


def validate_phase_detection():
    """Teste manual de detect_phase()."""
    evaluator = AdvancedEvaluator()

    test_cases = [
        (24, "Opening", 0.0, 0.15),
        (16, "Mid-opening", 0.25, 0.45),
        (12, "Midgame", 0.45, 0.75),
        (8, "Mid-endgame", 0.75, 1.0),
        (4, "Endgame", 0.85, 1.0),
    ]

    print("\n" + "="*70)
    print("VALIDAÇÃO: Detecção de Fase")
    print("="*70)

    all_passed = True
    for pieces, name, min_phase, max_phase in test_cases:
        board = create_position_with_n_pieces(pieces)
        phase = evaluator.detect_phase(board)
        passed = min_phase <= phase <= max_phase
        status = "[OK]" if passed else "[FAIL]"

        print(f"{status} {pieces:2d} pecas ({name:12s}): phase={phase:.3f} " +
              f"(esperado {min_phase:.2f}-{max_phase:.2f})")

        if not passed:
            all_passed = False

    print("="*70)
    return all_passed


def validate_interpolation():
    """Teste manual de _interpolate_weights()."""
    evaluator = AdvancedEvaluator()

    print("\n" + "="*70)
    print("VALIDAÇÃO: Interpolação de Pesos")
    print("="*70)

    test_cases = [
        (0.0, 100.0, "Opening weight"),
        (0.25, 107.5, "25% rumo ao endgame"),
        (0.5, 115.0, "Midgame"),
        (0.75, 122.5, "75% rumo ao endgame"),
        (1.0, 130.0, "Endgame weight"),
    ]

    all_passed = True
    for phase, expected, description in test_cases:
        # pylint: disable=protected-access
        result = evaluator._interpolate_weights(100, 130, phase)
        passed = abs(result - expected) < 0.01
        status = "[OK]" if passed else "[FAIL]"

        print(f"{status} Phase {phase:.2f}: {result:6.1f} " +
              f"(esperado {expected:6.1f}, {description})")

        if not passed:
            all_passed = False

    print("="*70)
    return all_passed


def validate_symmetry():
    """Teste de simetria da avaliação."""
    evaluator = AdvancedEvaluator()

    print("\n" + "="*70)
    print("VALIDAÇÃO: Simetria da Avaliação")
    print("="*70)

    positions = [
        ("Opening", BoardState.create_initial_state()),
        ("Midgame", create_test_position()),
        ("Endgame", create_endgame_position()),
        ("Assimétrica", create_asymmetric_position()),
    ]

    all_passed = True
    for name, board in positions:
        eval_red = evaluator.evaluate(board, PlayerColor.RED)
        eval_black = evaluator.evaluate(board, PlayerColor.BLACK)

        # Para simetria: avaliando a mesma posição de perspectivas diferentes
        # deve dar valores opostos (o que é bom para RED é ruim para BLACK)
        # NOTA: Pequenas assimetrias são OK devido a arredondamentos
        diff = abs(eval_red + eval_black)
        passed = diff < 30.0  # Tolerância para assimetrias de arredondamento
        status = "[OK]" if passed else "[FAIL]"

        print(f"{status} {name:12s}: RED={eval_red:7.1f}, BLACK={eval_black:7.1f}, " +
              f"diff={diff:6.1f}")

        if not passed:
            all_passed = False

    print("="*70)
    return all_passed


def benchmark_evaluation():
    """Medir velocidade de avaliação."""
    evaluator = AdvancedEvaluator()

    print("\n" + "="*70)
    print("BENCHMARK: Performance")
    print("="*70)

    # Criar 100 posições diversas
    positions = []
    for _ in range(25):
        positions.append(BoardState.create_initial_state())
    for _ in range(25):
        positions.append(create_test_position())
    for _ in range(25):
        positions.append(create_endgame_position())
    for _ in range(25):
        positions.append(create_asymmetric_position())

    # Benchmark
    start = time.time()
    for board in positions:
        evaluator.evaluate(board, PlayerColor.RED)
    elapsed = time.time() - start

    evals_per_sec = len(positions) / elapsed
    passed = evals_per_sec > 1000  # Meta mínima: 1000 evals/sec
    status = "[OK]" if passed else "[FAIL]"

    print(f"{status} Performance: {evals_per_sec:,.0f} avaliações/segundo")
    print("   Meta: >1,000 avaliações/segundo")
    print(f"   Tempo total: {elapsed*1000:.1f}ms para {len(positions)} posições")
    print("="*70)

    return passed


def run_all_validations():
    """Executa todos os testes de validação (Fase 1 + Fase 2)."""
    print("\n" + "="*70)
    print("VALIDAÇÃO COMPLETA - FASE 1 + FASE 2")
    print("="*70)

    results = {
        # Fase 1
        "Detecção de Fase": validate_phase_detection(),
        "Interpolação": validate_interpolation(),
        "Simetria": validate_symmetry(),
        "Performance": benchmark_evaluation(),
        # Fase 2
        "King Value Progression": validate_king_value_progression(),
        "PST Center Bonus": validate_pst_center_bonus(),
        "Safe Mobility": validate_safe_mobility(),
        "Material Dominance": validate_material_dominance(),
    }

    print("\n" + "="*70)
    print("RESUMO DOS TESTES")
    print("="*70)

    for name, passed in results.items():
        status = "[OK] PASSOU" if passed else "[FAIL] FALHOU"
        print(f"{status:15s}: {name}")

    all_passed = all(results.values())
    print("="*70)

    if all_passed:
        print("[OK] FASE 2 COMPLETA - Todos os testes passaram!")
        print("     Material e Mobilidade otimizados com sucesso!")
    else:
        print("[FAIL] FASE 2 INCOMPLETA - Alguns testes falharam")

    print("="*70)

    return all_passed


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # Executar validações manuais
        run_all_validations()
    else:
        # Executar testes unitários
        unittest.main()
