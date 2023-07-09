
"""
This script provides methods to map moves to IDs and vice versa,
allowing for efficient representation and lookup of moves in the game of checkers.
"""

MOVE_TO_ID = {}
ID_TO_MOVE = {}
ROTATED_ID_MAPPED = {}
ROTATED_MOVE_MAPED = {}
index = 0


@staticmethod
def is_on_board(row, col):
    """Check if given position (row and column) is within the board."""
    return 0 <= row < 10 and 0 <= col < 10


@staticmethod
def row_col_to_field(row, col):
    """Convert row and column into field representation."""
    return (row * 5 + col // 2) + 1



def add_in_dict_if_on_board(from_row, from_col, to_row, to_col, sign):
    """Add move to dictionary if both source and destination are within the board."""
    global index
    if is_on_board(from_row, from_col) and is_on_board(to_row, to_col):
        key = str(row_col_to_field(from_row, from_col)) + sign + str(row_col_to_field(to_row, to_col))
        if key not in MOVE_TO_ID:
            MOVE_TO_ID[key] = index
            index += 1
            return True

    return False


def add_moves(from_row, from_col):
    """
    Add all possible moves (excluding captures) from a given position.
    `capture` flag indicates whether to add capture moves (i.e., jumping over an opponent's piece).
    """
    for i in range(1, 10):
        add_in_dict_if_on_board(from_row, from_col, from_row - i, from_col - i, "-")  # upper left
        add_in_dict_if_on_board(from_row, from_col, from_row - i, from_col + i, "-")  # upper right
        add_in_dict_if_on_board(from_row, from_col, from_row + i, from_col - i, "-")  # lower left
        add_in_dict_if_on_board(from_row, from_col, from_row + i, from_col + i, "-")  # lower right

def add_captures(from_row, from_col):
    """
    Add all possible captures from a given position.
    `capture` flag indicates whether to add capture moves (i.e., jumping over an opponent's piece).
    """
    for i in range(2, 10):
        add_in_dict_if_on_board(from_row, from_col, from_row - i, from_col - i, "x")  # upper left
        add_in_dict_if_on_board(from_row, from_col, from_row - i, from_col + i, "x")  # upper right
        add_in_dict_if_on_board(from_row, from_col, from_row + i, from_col - i, "x")  # lower left
        add_in_dict_if_on_board(from_row, from_col, from_row + i, from_col + i, "x")  # lower right


def init_bitboard():
    """Initialize the bitboard by generating all possible moves and captures."""
    for row in range(10):
        for col in range((row+1)%2, 10, 2):
            add_moves(row, col)

    for row in range(10):
        for col in range((row+1)%2, 10, 2):
            add_captures(row, col)

    for move, id in MOVE_TO_ID.items():
        ID_TO_MOVE[id] = move

    for move, id in MOVE_TO_ID.items():
        if "x" in move:
            sign = "x"
        else:
            sign = "-"
        start_square, end_square = [int(val) for val in move.split(sign)]
        symmetric_start_square = 50 - start_square + 1
        symmetric_end_square = 50 - end_square + 1
        symmetric_move = str(symmetric_start_square) + sign + str(symmetric_end_square)
        ROTATED_MOVE_MAPED[move] = symmetric_move
        ROTATED_ID_MAPPED[id] = MOVE_TO_ID[symmetric_move]


init_bitboard()


