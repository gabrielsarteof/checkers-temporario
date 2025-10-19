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

# Board sem captura
b_no_capture = BoardState()
b_no_capture.set_piece(Piece(PlayerColor.RED, PieceType.NORMAL, Position(4, 3)))

print("Board COM captura:")
moves_cap = MoveGenerator.get_all_valid_moves(PlayerColor.RED, b_capture)
print(f"  RED moves: {len(moves_cap)}")
for m in moves_cap:
    print(f"    - {m.start} -> {m.end}, is_capture: {m.is_capture}")

mob_cap = e.evaluate_mobility(b_capture, PlayerColor.RED)
print(f"  Mobility: {mob_cap}")

print("\nBoard SEM captura:")
moves_no = MoveGenerator.get_all_valid_moves(PlayerColor.RED, b_no_capture)
print(f"  RED moves: {len(moves_no)}")
for m in moves_no:
    print(f"    - {m.start} -> {m.end}, is_capture: {m.is_capture}")

mob_no = e.evaluate_mobility(b_no_capture, PlayerColor.RED)
print(f"  Mobility: {mob_no}")

print(f"\nDiff: {mob_cap - mob_no} (esperado positivo)")
