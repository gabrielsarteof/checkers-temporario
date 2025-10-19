# FASE 12 - INTEGRAÇÃO COM MINIMAX

## Como Integrar Move Ordering com Minimax

Este documento mostra como integrar o sistema de move ordering da Fase 12 com seu algoritmo Minimax para maximizar a eficiência do alpha-beta pruning.

## Exemplo de Integração Completa

```python
from Checkers.evaluators.meninas_superpoderosas_evaluator import AdvancedEvaluator
from core.move_generator import MoveGenerator
from core.enums import PlayerColor

class MinimaxAI:
    def __init__(self, evaluator: AdvancedEvaluator):
        self.evaluator = evaluator
        self.nodes_explored = 0

    def get_best_move(self, board, color, max_depth):
        """
        Encontra o melhor movimento usando Minimax com alpha-beta pruning.
        """
        # Limpar killer moves no início de cada busca
        self.evaluator.clear_killer_moves()

        self.nodes_explored = 0
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # Gerar e ordenar movimentos
        moves = MoveGenerator.get_all_valid_moves(color, board)
        ordered_moves = self.evaluator.order_moves(moves, board, depth=max_depth)

        for move in ordered_moves:
            # Aplicar movimento
            board.apply_move(move)

            # Buscar recursivamente
            score = -self.minimax(
                board,
                max_depth - 1,
                -beta,
                -alpha,
                False,
                color
            )

            # Desfazer movimento
            board.undo_move(move)

            # Atualizar melhor movimento
            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

            # Armazenar killer move se causou cutoff
            if score >= beta:
                self.evaluator.store_killer_move(move, max_depth)
                self.evaluator.update_history_score(move, max_depth, True)
                break
            else:
                self.evaluator.update_history_score(move, max_depth, False)

        return best_move

    def minimax(
        self,
        board,
        depth,
        alpha,
        beta,
        maximizing,
        color
    ):
        """
        Minimax com alpha-beta pruning e move ordering (FASE 12).
        """
        self.nodes_explored += 1

        # Condições terminais
        if depth == 0:
            return self.evaluator.evaluate(board, color)

        # Verificar fim de jogo
        opponent_color = PlayerColor.BLACK if color == PlayerColor.RED else PlayerColor.RED
        current_color = color if maximizing else opponent_color

        moves = MoveGenerator.get_all_valid_moves(current_color, board)

        if not moves:
            # Sem movimentos = derrota
            return -10000 if maximizing else 10000

        # ================================================================
        # FASE 12: ORDENAR MOVIMENTOS ANTES DE EXPLORAR
        # ================================================================
        ordered_moves = self.evaluator.order_moves(moves, board, depth)

        if maximizing:
            max_eval = float('-inf')

            for move in ordered_moves:
                # Aplicar movimento
                board.apply_move(move)

                # Recursão
                eval_score = self.minimax(
                    board, depth - 1, alpha, beta, False, color
                )

                # Desfazer movimento
                board.undo_move(move)

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                # ========================================================
                # FASE 12: ATUALIZAR HEURISTICS QUANDO CUTOFF OCORRE
                # ========================================================
                if beta <= alpha:
                    # Beta cutoff - este move é BOM
                    self.evaluator.store_killer_move(move, depth)
                    self.evaluator.update_history_score(move, depth, True)
                    break  # Pruning
                else:
                    # Não causou cutoff
                    self.evaluator.update_history_score(move, depth, False)

            return max_eval

        else:  # Minimizing
            min_eval = float('inf')

            for move in ordered_moves:
                board.apply_move(move)

                eval_score = self.minimax(
                    board, depth - 1, alpha, beta, True, color
                )

                board.undo_move(move)

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    self.evaluator.store_killer_move(move, depth)
                    self.evaluator.update_history_score(move, depth, True)
                    break
                else:
                    self.evaluator.update_history_score(move, depth, False)

            return min_eval


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    from core.board_state import BoardState

    # Criar avaliador
    evaluator = AdvancedEvaluator()

    # Criar AI
    ai = MinimaxAI(evaluator)

    # Criar tabuleiro inicial
    board = BoardState.create_initial_state()

    # Buscar melhor movimento
    best_move = ai.get_best_move(board, PlayerColor.RED, max_depth=6)

    print(f"Melhor movimento: {best_move}")
    print(f"Nós explorados: {ai.nodes_explored}")
```

## Pontos Importantes

### 1. Limpar Killer Moves
```python
# No início de cada busca/jogo
evaluator.clear_killer_moves()
```

### 2. Ordenar Movimentos
```python
# ANTES de iterar sobre movimentos
ordered_moves = evaluator.order_moves(moves, board, depth)

for move in ordered_moves:
    # Explorar...
```

### 3. Armazenar Killer Moves
```python
# Quando ocorre beta-cutoff
if beta <= alpha:
    evaluator.store_killer_move(move, depth)
    break  # Pruning
```

### 4. Atualizar History Heuristic
```python
# Para TODOS os movimentos explorados
if caused_cutoff:
    evaluator.update_history_score(move, depth, True)
else:
    evaluator.update_history_score(move, depth, False)
```

## Benefícios Esperados

- **3-5x menos nós explorados** na mesma profundidade
- **1-2 plies mais profundos** no mesmo tempo
- **+100-200 Elo** em força de jogo
- Capturas sempre exploradas primeiro
- Movimentos táticos priorizados

## Debug e Análise

### Ver Prioridade de um Movimento
```python
info = evaluator.get_move_priority_info(move, board, depth)
print(info)

# Output:
# {
#     'move': '(5,0) -> (4,1)',
#     'is_capture': True,
#     'num_captured': 1,
#     'total_priority': 11000.0,
#     'breakdown': {'capture': 11000.0}
# }
```

### Desabilitar Move Ordering (para benchmark)
```python
# Sem move ordering
evaluator._move_ordering_enabled = False
nodes_without = run_benchmark()

# Com move ordering
evaluator._move_ordering_enabled = True
nodes_with = run_benchmark()

print(f"Redução: {(nodes_without - nodes_with) / nodes_without * 100:.1f}%")
```

## Manutenção Entre Jogos

```python
# Entre jogos diferentes
evaluator.clear_killer_moves()
evaluator.clear_history_scores()

# Entre buscas no MESMO jogo
evaluator.clear_killer_moves()  # Killers são específicos da árvore
# History pode ser mantido (geralmente benéfico)
```

## Performance

Move ordering adiciona overhead mínimo (<5%) mas resulta em:
- Muito menos nós explorados (30-70% de redução)
- Tempo total menor (speedup de 1.5-3x típico)
- Profundidade efetiva maior

**A profundidade extra vale MUITO mais que o pequeno overhead!**
