# Sistema de Registro de Avaliadores para Competição

## Visão Geral

Este módulo implementa um sistema extensível de registro de avaliadores para o jogo de damas, permitindo que múltiplas funções de avaliação compitam entre si. O sistema foi projetado seguindo o padrão **Strategy Pattern** com registro dinâmico, facilitando a adição de novos avaliadores sem modificação do código principal.

## Arquitetura

### Estrutura de Diretórios

```
evaluators/
├── __init__.py                           # Sistema de registro e factory
├── meninas_superpoderosas_evaluator.py   # Implementação de avaliador avançado
└── README.md                             # Documentação técnica
```

### Componentes Principais

#### 1. Sistema de Registro (`__init__.py`)

O sistema utiliza um dicionário `EVALUATOR_REGISTRY` que mapeia nomes de avaliadores para suas respectivas classes:

```python
EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {
    "Professor (Padrão)": PieceCountEvaluator,
    "Meninas Superpoderosas": AdvancedEvaluator,
}
```

**Funções Auxiliares:**
- `get_evaluator_names() -> list[str]`: Retorna lista de avaliadores disponíveis
- `get_evaluator_class(name: str) -> Type[BaseEvaluator]`: Retorna classe do avaliador
- `create_evaluator(name: str) -> BaseEvaluator`: Factory method para instanciar avaliadores
- `get_default_evaluator_name() -> str`: Retorna nome do avaliador padrão

**Tratamento de Erros:**
- Importações opcionais com try-except para evitar falhas se avaliador não disponível
- KeyError com mensagem descritiva se avaliador não encontrado no registro

#### 2. Interface BaseEvaluator

Todos os avaliadores implementam a interface abstrata `BaseEvaluator`:

```python
class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, board: BoardState, color: PlayerColor) -> float:
        """
        Avalia posição do tabuleiro do ponto de vista do jogador.

        Retorno:
            float > 0: Vantagem para 'color'
            float < 0: Desvantagem para 'color'
            float = 0: Posição equilibrada
        """
        pass
```

## Integração com GameManager

### Modificações Realizadas

O `GameManager` foi modificado para suportar avaliadores independentes por jogador:

**Parâmetros do Construtor:**
```python
def __init__(
    self,
    game_mode: GameMode = GameMode.HUMAN_VS_AI,
    difficulty: Difficulty = Difficulty.MEDIUM,
    red_evaluator_name: str = None,
    black_evaluator_name: str = None
):
```

**Inicialização de Jogadores:**
```python
def _initialize_players(self) -> None:
    red_evaluator = evaluators.create_evaluator(self.red_evaluator_name)
    black_evaluator = evaluators.create_evaluator(self.black_evaluator_name)

    self.red_player = AIPlayer(
        color=PlayerColor.RED,
        evaluator=red_evaluator,
        difficulty=self.difficulty
    )
    self.black_player = AIPlayer(
        color=PlayerColor.BLACK,
        evaluator=black_evaluator,
        difficulty=self.difficulty
    )
```

**Métodos de Configuração:**
- `set_red_evaluator(evaluator_name: str)`: Altera avaliador do jogador RED
- `set_black_evaluator(evaluator_name: str)`: Altera avaliador do jogador BLACK

## Interface Gráfica

### Componente EvaluatorSelector

Criado componente reutilizável `EvaluatorSelector` para seleção de avaliadores na UI:

**Características:**
- Criação dinâmica de botões baseada no registro de avaliadores
- Suporte a múltiplas instâncias (RED e BLACK)
- Título customizável por instância
- Visibilidade controlada por modo de jogo
- Feedback visual para seleção ativa

**Integração no `main.py`:**
```python
# Seletor para RED
self.red_evaluator_selector = EvaluatorSelector(
    x=left_column_x,
    y=evaluator_y,
    width=button_width,
    title="AVALIADOR RED",
    on_evaluator_change=self._on_red_evaluator_change,
    default_evaluator=evaluators.get_default_evaluator_name()
)

# Seletor para BLACK
self.black_evaluator_selector = EvaluatorSelector(
    x=left_column_x,
    y=black_evaluator_y,
    width=button_width,
    title="AVALIADOR BLACK",
    on_evaluator_change=self._on_black_evaluator_change,
    default_evaluator="Meninas Superpoderosas"
)
```

## Remoção de Dependências Hardcoded

### Antes (Enum Estático):
```python
class EvaluatorType(Enum):
    PROFESSOR = "Professor (Padrão)"
    ALUNO = "Aluno (Avançado)"
```

### Depois (Registro Dinâmico):
```python
EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {
    "Professor (Padrão)": PieceCountEvaluator,
    # Novos avaliadores adicionados aqui dinamicamente
}
```

**Vantagens:**
- Extensibilidade: Adicionar avaliadores sem modificar enums
- Flexibilidade: Avaliadores podem ser carregados dinamicamente
- Manutenibilidade: Separação de concerns (registro vs lógica)

## Avaliadores Implementados

### 1. PieceCountEvaluator (Professor)
**Localização:** `core/evaluation/piece_count_evaluator.py`

**Estratégia:** Avaliação baseada em contagem material simples

**Complexidade:** O(n) onde n = número de peças

**Fórmula:**
```
score = Σ(peças_normais × 1.0) + Σ(damas × 3.0)
```

### 2. AdvancedEvaluator (Meninas Superpoderosas)
**Localização:** `evaluators/meninas_superpoderosas_evaluator.py`

**Estratégia:** Heurística multi-componente com pesos adaptativos

**Complexidade:** O(n²) onde n = número de peças

**Componentes da Avaliação:**

1. **Vantagem Material** (40-50% do peso total)
   - Peças normais: 1.0 ponto
   - Damas: 3.0 pontos

2. **Controle Posicional** (15-20% do peso)
   - Centro do tabuleiro: +0.3
   - Linha de promoção: +0.5
   - Bordas: penalização

3. **Mobilidade** (10-15% do peso)
   - Movimentos simples: +0.1 por movimento
   - Capturas disponíveis: +0.3 por captura

4. **Estrutura de Peças** (10-15% do peso)
   - Peças adjacentes aliadas: +0.2 (proteção)
   - Agrupamento excessivo: penalização

5. **Ameaças de Captura** (15-20% do peso)
   - Peças ameaçadas: -0.5
   - Peças protegidas: +0.3

6. **Progresso de Promoção** (5-10% do peso)
   - Proximidade da linha de promoção: variável

**Ajuste Dinâmico por Fase:**
```python
def _get_game_phase(self, board: BoardState) -> str:
    total_pieces = len(board.get_all_pieces())
    if total_pieces > 16:
        return "opening"    # Peso maior em posição
    elif total_pieces > 8:
        return "midgame"    # Balanceado
    else:
        return "endgame"    # Peso maior em promoção
```

## Adicionando Novos Avaliadores

### Passo 1: Implementação

Criar arquivo no diretório `evaluators/`:

```python
"""Descrição do avaliador."""

from core.evaluation.base_evaluator import BaseEvaluator
from core.board_state import BoardState
from core.enums import PlayerColor

class NovoAvaliador(BaseEvaluator):
    """Implementação de estratégia XYZ."""

    def evaluate(self, board: BoardState, color: PlayerColor) -> float:
        # Implementação da heurística
        return score
```

### Passo 2: Registro

Adicionar ao `__init__.py`:

```python
try:
    from evaluators.novo_avaliador import NovoAvaliador
    NOVO_AVALIADOR_AVAILABLE = True
except ImportError:
    NOVO_AVALIADOR_AVAILABLE = False

if NOVO_AVALIADOR_AVAILABLE:
    EVALUATOR_REGISTRY["Novo Avaliador"] = NovoAvaliador
```

### Passo 3: Teste

Executar script de verificação:

```bash
python test_competition.py
```

## Testes Realizados

### Test Suite: `test_competition.py`

**Casos de Teste:**
1. Listagem de avaliadores disponíveis
2. Criação de jogo IA vs IA com avaliadores diferentes
3. Verificação de configuração correta
4. Mudança dinâmica de avaliadores
5. Integridade dos jogadores AI

**Resultado:** ✅ Todos os testes passaram com sucesso

## Decisões de Design

### 1. Por que Registro Dinâmico?
- **Extensibilidade:** Novos avaliadores sem modificar código existente (Open/Closed Principle)
- **Testabilidade:** Fácil mockar avaliadores para testes
- **Manutenibilidade:** Reduz acoplamento entre componentes

### 2. Por que Separar RED e BLACK?
- **Flexibilidade:** Permite competições assimétricas
- **Experimentação:** Testar variações do mesmo avaliador
- **Análise:** Comparar performance em cores diferentes

### 3. Por que UI Dinâmica?
- **Sincronização:** UI sempre reflete avaliadores disponíveis
- **Usabilidade:** Não requer recompilação para adicionar avaliadores
- **Escalabilidade:** Suporta número arbitrário de avaliadores

## Complexidade Computacional

### Análise por Componente:

| Componente | Complexidade | Justificativa |
|------------|--------------|---------------|
| Registro | O(1) | Dict lookup |
| Factory | O(1) | Instanciação de classe |
| UI Update | O(n) | n = número de avaliadores |
| Material | O(n) | n = número de peças |
| Mobilidade | O(n×m) | n = peças, m = movimentos médios |
| Estrutura | O(n²) | Comparação entre peças |

**Otimizações Possíveis:**
- Cache de avaliações para estados repetidos (Transposition Table)
- Avaliação incremental (calcular apenas diferenças)
- Poda de componentes menos relevantes em fases específicas

## Limitações e Trabalhos Futuros

### Limitações Atuais:
1. Avaliadores devem estar em arquivos únicos (não podem usar módulos externos)
2. Sem suporte a hot-reload (requer restart para novos avaliadores)
3. Configuração de pesos hardcoded (não há tuning automático)

### Melhorias Propostas:
1. Sistema de plugins com carregamento dinâmico em runtime
2. Interface para tuning de parâmetros via UI
3. Histórico de partidas e estatísticas de performance
4. Suporte a torneios round-robin automáticos
5. Exportação de logs para análise externa

## Conclusão

O sistema implementado fornece uma arquitetura extensível e bem estruturada para competição entre diferentes estratégias de avaliação no jogo de damas. A separação de concerns, uso de padrões de design, e interface dinâmica garantem que novos avaliadores possam ser adicionados e testados com mínimo esforço, mantendo a integridade do código base.

## Referências

- Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*
- Russell, S., Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.)
- Shannon, C. (1950). "Programming a Computer for Playing Chess"
