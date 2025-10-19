import sys
sys.path.insert(0, '.')

from evaluators.meninas_superpoderosas_evaluator import AdvancedEvaluator
from core.board_state import BoardState
from core.piece import Piece
from core.position import Position
from core.enums import PlayerColor, PieceType
from core.move_generator import MoveGenerator

e = AdvancedEvaluator()

# Board com captura
b_capture = BoardState()
b_capture.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, Position(4, 3)))
b_capture.set_piece(Piece(PlayerColor.BLACK, PieceType.NORMAL, Position(3, 2)))

print("Analisando mobility para board COM captura:")

# Gerar moves
player_moves = MoveGenerator.get_all_valid_moves(PlayerColor.RED, b_capture)
opp_moves = MoveGenerator.get_all_valid_moves(PlayerColor.BLACK, b_capture)

print(f"RED moves: {len(player_moves)}")
for m in player_moves:
    print(f"  - {m.start} -> {m.end}, is_capture: {m.is_capture}")

print(f"BLACK moves: {len(opp_moves)}")
for m in opp_moves:
    print(f"  - {m.start} -> {m.end}, is_capture: {m.is_capture}")

# Threatened squares
threatened = e._get_threatened_squares(opp_moves)
print(f"Threatened squares by BLACK: {threatened}")

# Classificar moves
player_total = len(player_moves)
player_safe = 0
player_captures = 0

for move in player_moves:
    if move.is_capture:
        player_captures += 1
    if move.end not in threatened:
        player_safe += 1

print(f"\nRED stats:")
print(f"  Total: {player_total}")
print(f"  Captures: {player_captures}")
print(f"  Safe: {player_safe}")

# Calcular mobility
player_mobility = (
    player_total * e.MOVE_BASE_VALUE +
    player_safe * e.SAFE_MOVE_BONUS +
    player_captures * e.CAPTURE_MOVE_VALUE
)

print(f"  Mobility (before opponent): {player_mobility}")

# Oponente
player_threatened = e._get_threatened_squares(player_moves)
opp_total = len(opp_moves)
opp_safe = 0
opp_captures = 0

for move in opp_moves:
    if move.is_capture:
        opp_captures += 1
    if move.end not in player_threatened:
        opp_safe += 1

opp_mobility = (
    opp_total * e.MOVE_BASE_VALUE +
    opp_safe * e.SAFE_MOVE_BONUS +
    opp_captures * e.CAPTURE_MOVE_VALUE
)

print(f"\nBLACK stats:")
print(f"  Total: {opp_total}")
print(f"  Captures: {opp_captures}")
print(f"  Safe: {opp_safe}")
print(f"  Mobility: {opp_mobility}")

mob_diff = player_mobility - opp_mobility
print(f"\nMobility diff (before weight): {mob_diff}")

phase = e.detect_phase(b_capture)
mob_weight = e._interpolate_weights(e.MOBILITY_WEIGHT_OPENING, e.MOBILITY_WEIGHT_ENDGAME, phase)
print(f"Phase: {phase}, Mobility weight: {mob_weight}")
print(f"Final mobility: {mob_diff * mob_weight}")
