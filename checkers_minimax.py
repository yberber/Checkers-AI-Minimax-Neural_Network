import time

import numpy as np

from checkers_engine import CheckersGameState
import copy

DEPTH = 0
counter = 0
current_time = 0
move_counter = 0
current_player_white = None

start_time = time.time()



def find_move_minimax_alpha_beta(gs: CheckersGameState, search_depth: int, alpha: int, beta: int, is_maximizing_player: bool, is_recursive_call=False, quiescence_search=True):
    """
    Finds the best move using Minimax algorithm with Alpha-Beta pruning and Quiescence search.

    Args:
        gs (game state object): The current game state.
        search_depth (int): Depth to which minimax algorithm should recursively evaluate.
        alpha (int): The best (highest) score found for the maximizing player.
        beta (int): The best (lowest) score found for the minimizing player.
        is_maximizing_player (bool): Is the current player maximizing player?
        is_recursive_call (bool): Is this function call a recursive call?
        quiescence_search (bool): Is quiescence search applied?

    Returns:
        int: The score of the best move.
        object: The best move itself.
    """
    global counter
    counter += 1
    # Quiescence search: when depth is exhausted, only capturing moves are considered
    if search_depth <= 0:
        if not quiescence_search:
            return evaluate_board(gs.board), None
        valid_moves = gs.fetch_only_possible_captures()
        if not len(valid_moves) > 0:
            return evaluate_board(gs.board), None
    else:
        valid_moves = gs.fetch_all_possible_moves()
        if len(valid_moves) == 0:
            return -100 if gs.white_to_move else 100, None

    # If not a recursive call,
    # Randomize moves to add variation in games when multiple equally good moves exist
    if not is_recursive_call:
        np.random.shuffle(valid_moves)

    best_move = None

    # Maximizing player
    if is_maximizing_player:
        max_score = -np.inf
        for move in valid_moves:
            gs.make_minimax_move(move)
            score, _ = find_move_minimax_alpha_beta(gs, search_depth - 1, alpha, beta, False, True, quiescence_search)
            gs.undo_move()

            if score > max_score:
                max_score = score
                alpha = max(alpha, max_score)
                # Beta cut-off
                if alpha >= beta:
                    break
                if not is_recursive_call:
                    best_move = move
        return max_score, best_move

    else:
        min_score = np.inf
        for move in valid_moves:
            gs.make_minimax_move(move)
            score, _ = find_move_minimax_alpha_beta(gs, search_depth - 1, alpha, beta, True, True, quiescence_search)
            gs.undo_move()

            if score < min_score:
                min_score = score
                beta = min(beta, min_score)
                # Alpha cut-off
                if beta <= alpha:
                    break
                if not is_recursive_call:
                    best_move = move
        return min_score, best_move


def find_best_move_minimax(gs: CheckersGameState, search_depth=8, depth_reduction_threshold=None):
    """
    Uses the Minimax algorithm with alpha-beta pruning to find the optimal move for the current game state.

    Parameters:
    gs (GameState): The current game state.
    search_depth (int, optional): The depth to which the game tree will be searched. Defaults to 8.
    depth_reduction_threshold (int, optional): The threshold of valid moves count at which the search depth will be
                                               reduced by 1 to save computation time. Defaults to None.

    Returns:
    next_move (Move): The optimal move found by the Minimax algorithm.
    """
    global DEPTH, counter, current_time, move_counter, current_player_white
    counter = 0
    if (depth_reduction_threshold is not None) and (len(gs.valid_moves) >= depth_reduction_threshold) and (search_depth > 1):
        search_depth -= 1
    DEPTH = search_depth
    current_time = time.time()
    current_player_white = gs.white_to_move
    valid_moves = gs.compute_if_needed_and_get_single_valid_moves()
    if len(valid_moves) == 1:
        best_move = valid_moves[0]
        return best_move
    score, next_move = find_move_minimax_alpha_beta(copy.deepcopy(gs), search_depth, -255, 255, gs.white_to_move)
    move_counter += 1
    print(f"turn: {'white' if gs.white_to_move else 'black'}, counter: {counter}, score: {score}"
          f", used time: {time.time() - current_time}, used total time: {time.time() - start_time}, "
          f"total generated move count: {move_counter}, move: {[str(m) for m in next_move] if type(next_move) is list else str(next_move)}")

    return next_move


piece_score = {"k": 3, "p": 1, "-": 0}
def score_material(board):
    """
    Evaluates the material score of a given checkers board state.
    Positive scores are good for the white while negatives for the black

    Parameters:
    board (list): A 2D list representing the checkers board.

    Returns:
    score (int): The material score of the board state.
    """
    score = 0
    for row in board:
        for square in row:
            if square[0] == 'w':
                score += piece_score[square[1]]
            elif square[0] == 'b':
                score -= piece_score[square[1]]

    return score


white_color = 'w'
black_color = 'b'
king_sign = 'k'
pawn_sign = 'p'
piece_score = {king_sign: 3, pawn_sign: 1}
color_multiplier = {black_color: -1, white_color: 1}
def evaluate_board(board):
    score = 0
    for row_index, row in enumerate(board):
        for square in row[(row_index + 1) % 2::2]:
            if square[1] == pawn_sign:
                score += piece_score[pawn_sign] * color_multiplier[square[0]]
                score += (10 - row_index)/8 if white_color == square[0] else -(row_index + 1)/8
            elif square[1] == king_sign:
                score += piece_score[pawn_sign] * color_multiplier[square[0]]
    return score
