"""
Sistema de Registro de Avaliadores
===================================

Este módulo fornece um registro centralizado de todos os avaliadores disponíveis.

Para adicionar um novo avaliador:
1. Crie um arquivo .py neste diretório (ex: meu_evaluator.py)
2. Implemente uma classe que herde de BaseEvaluator
3. Adicione o avaliador ao EVALUATOR_REGISTRY abaixo

Exemplo:
    from evaluators.meu_evaluator import MeuEvaluator

    EVALUATOR_REGISTRY = {
        "Meu Avaliador": MeuEvaluator,
        ...
    }
"""

from typing import Dict, Type
from core.evaluation.base_evaluator import BaseEvaluator
from core.evaluation.piece_count_evaluator import PieceCountEvaluator

# Import dos avaliadores customizados
    
try:
    from evaluators.sarte_evaluator import AdvancedEvaluatorSarte
    SARTE_AVAILABLE = True
except ImportError:
    SARTE_AVAILABLE = False
    print("[AVISO] Sarte evaluator não disponível")

try:
    from evaluators.meninas_superpoderosas_evaluator import MeninasSuperPoderosasEvaluator
    MENINAS_SUPERPODEROSAS_AVAILABLE = True
except ImportError:
    MENINAS_SUPERPODEROSAS_AVAILABLE  = False
    print("[AVISO] Meninas Super Poderosas evaluator não disponível")

try:
    from evaluators.openingbook_evaluator import OpeningBookEvaluator
    OPENINGBOOK_AVAILABLE = True
except ImportError:
    OPENINGBOOK_AVAILABLE  = False
    print("[AVISO] Opening Book evaluator não disponível")


# ============================================================================
# REGISTRO DE AVALIADORES
# ============================================================================

EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {
    "Professor (Padrão)": PieceCountEvaluator,
}

# Adicionar avaliadores customizados se disponíveis
if MENINAS_SUPERPODEROSAS_AVAILABLE:
    EVALUATOR_REGISTRY["Meninas Superpoderosas"] = MeninasSuperPoderosasEvaluator
if SARTE_AVAILABLE:
    EVALUATOR_REGISTRY["Sarte"] = AdvancedEvaluatorSarte
if OPENINGBOOK_AVAILABLE:
    EVALUATOR_REGISTRY["Opening Book"] = OpeningBookEvaluator


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def get_evaluator_names() -> list[str]:
    """
    Retorna lista de nomes de todos os avaliadores disponíveis.

    Returns:
        Lista de nomes dos avaliadores
    """
    return list(EVALUATOR_REGISTRY.keys())


def get_evaluator_class(name: str) -> Type[BaseEvaluator]:
    """
    Retorna a classe do avaliador pelo nome.

    Args:
        name: Nome do avaliador

    Returns:
        Classe do avaliador

    Raises:
        KeyError: Se o avaliador não existe
    """
    if name not in EVALUATOR_REGISTRY:
        raise KeyError(f"Avaliador '{name}' não encontrado. Disponíveis: {get_evaluator_names()}")

    return EVALUATOR_REGISTRY[name]


def create_evaluator(name: str) -> BaseEvaluator:
    """
    Cria uma instância do avaliador pelo nome.

    Args:
        name: Nome do avaliador

    Returns:
        Instância do avaliador

    Raises:
        KeyError: Se o avaliador não existe
    """
    evaluator_class = get_evaluator_class(name)
    return evaluator_class()


def get_default_evaluator_name() -> str:
    """
    Retorna o nome do avaliador padrão.

    Returns:
        Nome do avaliador padrão
    """
    return "Professor (Padrão)"


# ============================================================================
# INFORMAÇÕES
# ============================================================================

def print_available_evaluators():
    """Imprime todos os avaliadores disponíveis."""
    print("\n" + "=" * 70)
    print("AVALIADORES DISPONÍVEIS")
    print("=" * 70)

    for i, name in enumerate(get_evaluator_names(), 1):
        evaluator_class = EVALUATOR_REGISTRY[name]
        print(f"{i}. {name}")
        if hasattr(evaluator_class, "__doc__") and evaluator_class.__doc__:
            doc = evaluator_class.__doc__.strip().split('\n')[0]
            print(f"   {doc}")

    print("=" * 70)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'EVALUATOR_REGISTRY',
    'get_evaluator_names',
    'get_evaluator_class',
    'create_evaluator',
    'get_default_evaluator_name',
    'print_available_evaluators'
]
