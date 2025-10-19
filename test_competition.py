"""Script de teste para competição entre avaliadores."""

import evaluators
from core.game_manager import GameManager
from core.enums import GameMode, Difficulty

def test_competition():
    """Testa competição entre avaliadores."""
    print("=" * 70)
    print("TESTE DE COMPETIÇÃO ENTRE AVALIADORES")
    print("=" * 70)

    # Listar avaliadores disponíveis
    print("\nAvaliadores disponíveis:")
    for i, name in enumerate(evaluators.get_evaluator_names(), 1):
        print(f"  {i}. {name}")

    # Criar jogo IA vs IA
    print("\nCriando jogo IA vs IA...")
    print("  RED: Professor (Padrão)")
    print("  BLACK: Meninas Superpoderosas")

    game = GameManager(
        game_mode=GameMode.AI_VS_AI,
        difficulty=Difficulty.MEDIUM,
        red_evaluator_name="Professor (Padrão)",
        black_evaluator_name="Meninas Superpoderosas"
    )

    # Verificar configuração
    print("\nVerificando configuração:")
    print(f"  Modo: {game.game_mode.value}")
    print(f"  Dificuldade: {game.difficulty.value}")
    print(f"  RED Evaluator: {game.red_evaluator_name}")
    print(f"  BLACK Evaluator: {game.black_evaluator_name}")

    # Verificar jogadores
    print("\nJogadores:")
    if game.red_player:
        print(f"  RED: {game.red_player.name}")
    else:
        print(f"  RED: Humano")

    if game.black_player:
        print(f"  BLACK: {game.black_player.name}")
    else:
        print(f"  BLACK: Humano")

    # Testar mudança de avaliadores
    print("\nTestando mudança de avaliadores...")
    game.set_red_evaluator("Meninas Superpoderosas")
    game.set_black_evaluator("Professor (Padrão)")

    print(f"  RED Evaluator agora: {game.red_evaluator_name}")
    print(f"  BLACK Evaluator agora: {game.black_evaluator_name}")

    print("\n" + "=" * 70)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    test_competition()
