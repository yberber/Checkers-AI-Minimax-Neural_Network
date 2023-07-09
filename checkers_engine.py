import numpy as np
import move_id_mapper
import copy


class CheckersGameState:
    """
    A class representing the game state of a checkers game.

    Attributes:
    grid_shape (tuple): The shape of the game board grid.
    row_count (int): The number of rows on the game board.
    column_count (int): The number of columns on the game board.
    channels (int): The number of distinct pieces or states each square on the game board can have.
    action_size (int): The number of possible moves at any point in the game.
    """

    grid_shape = (10, 10)
    row_count = 10
    column_count = 10
    channels = 7
    action_size = len(move_id_mapper.MOVE_TO_ID)
    def __init__(self):
        # board is a 10x10 2d list, each element of the list has 2 characters.
        # the first character represents the color of the piece, 'b' or 'w'
        # the second character represents the type of the piece, p, k
        # "--" - represents an empty space with no piece
        self.valid_moves = []
        self.single_valid_moves = []
        self.board = [
            ["--", "bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp"],
            ["bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp", "--"],
            ["--", "bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp"],
            ["bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp", "--"],

            ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--"],

            ["--", "wp", "--", "wp", "--", "wp", "--", "wp", "--", "wp"],
            ["wp", "--", "wp", "--", "wp", "--", "wp", "--", "wp", "--"],
            ["--", "wp", "--", "wp", "--", "wp", "--", "wp", "--", "wp"],
            ["wp", "--", "wp", "--", "wp", "--", "wp", "--", "wp", "--"]
        ]


        self.move_functions = {'p': self._get_pawn_moves, 'k': self._get_king_moves}
        self.capture_functions = {'p': self._get_pawn_captures, 'k': self._get_king_captures}

        self.white_to_move = True
        self.move_log = []
        self.is_capturing = False
        self.capture_index = 0
        self.blocked_squares = []
        self.move_count = 0

    def reset(self):
        """
        Resets the game state to the initial state.
        """
        self.move_count = 0
        self.capture_index = 0
        self.valid_moves.clear()
        self.is_capturing = False
        self.white_to_move = True
        self.blocked_squares.clear()
        self.move_log.clear()
        self.single_valid_moves.clear()
        self.board = [
            ["--", "bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp"],
            ["bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp", "--"],
            ["--", "bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp"],
            ["bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp", "--"],

            ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--"],

            ["--", "wp", "--", "wp", "--", "wp", "--", "wp", "--", "wp"],
            ["wp", "--", "wp", "--", "wp", "--", "wp", "--", "wp", "--"],
            ["--", "wp", "--", "wp", "--", "wp", "--", "wp", "--", "wp"],
            ["wp", "--", "wp", "--", "wp", "--", "wp", "--", "wp", "--"]
        ]




    def make_move(self, move, simulation=False):
        """
        Execute a given move in the game state.

        Args:
        move (Move): The move to be executed.
        simulation (bool): Optional; whether the move is part of a simulation or not. Default is False.
        """
        self.board[move.start_row][move.start_col] = "--"
        self.board[move.end_row][move.end_col] = move.piece_moved
        if move.captured_piece != "--":
            row, col = move.captured_piece_pos
            self.board[row][col] = "--"
        self.move_log.append(move)  # log the move so we can undo it later

        if not simulation:
            if not self.is_capturing:
                self.switch_turns()
            else:
                self.capture_index += 1
                if self.capture_index >= len(self.valid_moves[0]):
                    self.switch_turns()
                else:
                    if move.piece_moved[1] == "k":
                        self.blocked_squares.append(move.captured_piece_pos)
                    for index in range(len(self.valid_moves)-1, -1, -1):
                        if self.move_log[-1] != self.valid_moves[index][self.capture_index-1]:
                            self.valid_moves.pop(index)

    def make_minimax_move(self, single_move_or_move_list):
        """
        Execute a single move or a sequence of moves.
        Change the turn after execution

        Args:
        single_move_or_move_list (Move or list of Move): A single move or a list of moves to be executed.
        """
        if type(single_move_or_move_list) is not list:
            self.make_move(single_move_or_move_list, simulation=True)
        else:
            for move in single_move_or_move_list:
                self.make_move(move, simulation=True)
        self.switch_turns()

    def make_move_extended(self, single_move_or_move_list):
        """
        Executes a single move or a sequence of moves. It's an extended version of the execute_move method.
        Change the turn automatically by the make_move method

        Args:
        single_move_or_move_list (Move or list of Move): A single move or a list of moves to be executed.
        """
        if type(single_move_or_move_list) is not list:
            self.make_move(single_move_or_move_list)
        else:
            for move in single_move_or_move_list:
                self.make_move(move)

    def print_board(self):
        """
        Prints the current state of the game board.
        """
        for row in range(10):
            for col in range(10):
                print(f" {self.board[row][col]}", end="")
            print()
        print()

    def get_turn_count_by_log(self) -> int:
        """
        Counts the number of turns taken by counting distinct player moves.

        Returns:
        int: The number of turns taken so far.
        """
        is_whites_move = None
        move_cnt = 0
        for move in self.move_log:
            if is_whites_move != move.is_white:
                is_whites_move = move.is_white
                move_cnt += 1
        if len(self.move_log):
            if self.move_log[-1] == self.white_to_move:
                move_cnt-=1
        return move_cnt

    def undo_move(self, only_one=False):
        if len(self.move_log):  # make sure that there is a move to undo
            while(True):
                move = self.move_log.pop()
                self.board[move.start_row][move.start_col] = move.piece_moved
                self.board[move.end_row][move.end_col] = "--"
                if move.captured_piece != "--":
                    self.board[move.captured_piece_pos[0]][move.captured_piece_pos[1]] = move.captured_piece
                if only_one:
                    return

                if not len(self.move_log) or move.is_white != self.move_log[-1].is_white:
                    break
            if (self.capture_index > 0):
                self.white_to_move = not self.white_to_move
            else:
                self.move_count-=1
            self.switch_turns(increment_move_count=False)


    def compute_if_needed_and_get_single_valid_moves(self):
        """
         Calculate and return single valid moves if they are not already computed.
         If capture is in progress, it computes the next capture move.

         Returns:
         list: A list of single valid moves.
         """
        if len(self.valid_moves) == 0:
            self.valid_moves = self.fetch_all_possible_moves()

        self.single_valid_moves = []
        if self.is_capturing:
            for move_sequence in self.valid_moves:
                self.single_valid_moves.append(move_sequence[self.capture_index])
        else:
            self.single_valid_moves = self.valid_moves
        return self.single_valid_moves

    def fetch_valid_moves(self):
        """
        Fetch all valid moves if they are not already computed.

        Returns:
        list: A list of all valid moves.
        """
        if len(self.valid_moves) == 0:
            self.valid_moves = self.fetch_all_possible_moves()
        return  self.valid_moves

    def fetch_valid_moves_for_piece(self, sq_selected):
        """
        Fetch valid moves for the piece at the selected square.

        Args:
        sq_selected (tuple): The selected square (row, column).

        Returns:
        list: A list of valid moves for the selected piece.
        """
        valid_moves_for_selected_piece = []
        for move in self.single_valid_moves:
            if sq_selected == (move.start_row, move.start_col):
                valid_moves_for_selected_piece.append(move)
        return valid_moves_for_selected_piece

    def fetch_all_possible_moves(self):
        """
        Compute and return all possible moves for the current player.
        This method computes capturing moves and non-capturing moves separately.

        Returns:
        list: A list of all possible moves.
        """
        moves = []
        moves_with_captures = []
        for row in range(len(self.board)):  # number of rows
            for col in range(len(self.board[row])):  # number of cols in given row
                turn = self.board[row][col][0]
                if turn == "w" and self.white_to_move or turn == "b" and not self.white_to_move:
                    piece = self.board[row][col][1]
                    self.capture_functions[piece](row, col, moves_with_captures)
                    self.move_functions[piece](row, col, moves)
        if len(moves_with_captures) != 0:
            return moves_with_captures
        return moves

    def fetch_only_possible_captures(self):
        """
        Compute and return only the possible capture moves for the current player.

        Returns:
        list: A list of possible capture moves.
        """
        moves_with_captures = []
        for row in range(len(self.board)):  # number of rows
            for col in range(len(self.board[row])):  # number of cols in given row
                turn = self.board[row][col][0]
                if turn == "w" and self.white_to_move or turn == "b" and not self.white_to_move:
                    piece = self.board[row][col][1]
                    self.capture_functions[piece](row, col, moves_with_captures)
        return moves_with_captures



    def _get_pawn_moves(self, row, col, moves):
        """
        Compute and store the non-capturing moves of a man from the given square.

        Args:
        row (int): The row of the man.
        col (int): The column of the man.
        moves (list): A list where the valid moves will be stored.
        """
        directions = self.get_move_directions(self.board[row][col])
        for d in directions:
            end_row = row + d[0]
            end_col = col + d[1]
            if self.is_within_board(end_row, end_col) and self.board[end_row][end_col] == "--":
                moves.append(Move((row, col), (end_row, end_col), self.board, self.white_to_move))

    def _get_pawn_captures(self, row, col, moves_with_captures, enemy_color=None, found_capture_list=[]):
        """
        Recursively compute and store the capturing moves of a man from the given square.

        Args:
        row (int): The row of the man.
        col (int): The column of the man.
        moves_with_captures (list): A list where the valid capture moves will be stored.
        enemy_color (str, optional): The color of the enemy pieces. Defaults to the color of the opponent of the current player.
        found_capture_list (list, optional): A list used in the recursion to store the intermediate capture sequences.
        """
        directions = self.get_capture_directions()
        if enemy_color is None:
            enemy_color = "b" if self.white_to_move else "w"
        any_found = False
        for d in directions:
            end_row = row + 2 * d[0]
            end_col = col + 2 * d[1]
            if self.is_within_board(end_row, end_col) and self.board[end_row][end_col] == "--":
                next_row = row + d[0]
                next_col = col + d[1]
                piece = self.board[next_row][next_col]
                if piece[0] == enemy_color and (next_row, next_col):
                    any_found = True
                    self.is_capturing = True
                    # self.blocked_squares.append((end_row, end_col))
                    tmp_move = Move((row, col), (end_row, end_col), self.board, self.white_to_move,
                                    captured_piece=piece, captured_piece_pos=(next_row, next_col))
                    found_capture_list.append(tmp_move)
                    self.make_move(tmp_move, simulation=True)
                    self._get_pawn_captures(end_row, end_col, moves_with_captures,
                                            enemy_color=enemy_color, found_capture_list=found_capture_list)
                    found_capture_list.pop()
                    # self.blocked_squares.pop()
                    self.undo_move(only_one=True)

        self.add_in_moves_with_captures(any_found, found_capture_list, moves_with_captures)

    # Get all the king moves for the king located at row, col and add these moves to the list
    def _get_king_moves(self, row, col, moves):
        """
        Compute and store the non-capturing moves of a king from the given square.

        Args:
        row (int): The row of the king.
        col (int): The column of the king.
        moves (list): A list where the valid moves will be stored.
        """
        directions = self.get_move_directions(self.board[row][col])
        for d in directions:
            for i in range(1, 10):
                end_row = row + d[0] * i
                end_col = col + d[1] * i
                if self.is_within_board(end_row, end_col):  # on board
                    end_piece = self.board[end_row][end_col]
                    if end_piece == "--":  # empty space valid
                        moves.append(Move((row, col), (end_row, end_col), self.board, self.white_to_move))
                    else:
                        break
                else:
                    break

    def _get_king_captures(self, row, col, moves_with_captures, enemy_color=None, found_capture_list=[]):
        """
        Recursively compute and store the capturing moves of a king from the given square.

        Args:
        row (int): The row of the king.
        col (int): The column of the king.
        moves_with_captures (list): A list where the valid capture moves will be stored.
        enemy_color (str, optional): The color of the enemy pieces. Defaults to the color of the opponent of the current player.
        found_capture_list (list, optional): A list used in the recursion to store the intermediate capture sequences.
        """
        directions = self.get_capture_directions()
        if enemy_color is None:
            enemy_color = "b" if self.white_to_move else "w"
        any_found = False
        for d in directions:
            for i in range(1, 9):
                capture_pos = ((row + d[0] * i), (col + d[1] * i))
                if capture_pos in self.blocked_squares:
                    break
                if self.is_within_board(capture_pos[0], capture_pos[1]):
                    piece = self.board[capture_pos[0]][capture_pos[1]]
                    if piece == "--":
                        continue
                    elif piece[0] == enemy_color :
                        for j in range(i + 1, 10):
                            end_row = row + d[0] * j
                            end_col = col + d[1] * j
                            if (end_row, end_col) in self.blocked_squares:
                                break
                            if self.is_within_board(end_row, end_col) and self.board[end_row][end_col] == "--":
                                any_found = True
                                self.is_capturing = True
                                tmp_move = Move((row, col), (end_row, end_col), self.board, self.white_to_move, captured_piece=piece, captured_piece_pos=capture_pos)
                                found_capture_list.append(tmp_move)
                                self.blocked_squares.append(capture_pos)
                                self.make_move(tmp_move, simulation=True)
                                self._get_king_captures(end_row, end_col, moves_with_captures, enemy_color=enemy_color,
                                                        found_capture_list=found_capture_list)
                                found_capture_list.pop()
                                self.blocked_squares.pop()
                                self.undo_move(True)
                            else:
                                break
                        break
                    else:
                        break
                else:
                    break

        self.add_in_moves_with_captures(any_found, found_capture_list, moves_with_captures)


    def switch_turns(self, increment_move_count=True):
        """
        Change the turn of the current game state, effectively switching which player is to move.
        Also, checks for potential promotion of a piece to king.

        Args:
        increment_move_count (bool, optional): Whether to increment the move count or not. Default is True.
        """
        self.capture_index = 0
        self.valid_moves = []
        self.is_capturing = False
        self.white_to_move = not self.white_to_move  # switch turns
        self.blocked_squares.clear()
        # man promotion to king
        if len(self.move_log):
            last_move = self.move_log[-1]
            if last_move.is_pawn_promotion:
                self.board[last_move.end_row][last_move.end_col] = last_move.piece_moved[0] + "k"
        if increment_move_count:
            self.move_count+=1

    @staticmethod
    def add_in_moves_with_captures(any_found, found_capture_list, moves_with_captures):
        """
        Update the moves_with_captures list if a capturing move sequence is found.

        Args:
        any_found (bool): Indicates whether any capture sequences were found.
        found_capture_list (List[Move]): List of found capturing move sequences.
        moves_with_captures (List[List[Move]]): List to be updated with the found capturing move sequences.
        """
        if not any_found and len(found_capture_list) > 0:
            if len(moves_with_captures) == 0:
                moves_with_captures.append(found_capture_list.copy())
            elif len(moves_with_captures[0]) < len(found_capture_list):
                moves_with_captures.clear()
                moves_with_captures.append(found_capture_list.copy())
            elif len(moves_with_captures[0]) == len(found_capture_list):
                moves_with_captures.append(found_capture_list.copy())



    @staticmethod
    def is_within_board(row, col):
        """
        Check if a given position (row, col) is within the game board.

        Args:
        row (int): Row of the position to be checked.
        col (int): Column of the position to be checked.

        Returns:
        bool: True if position is within the board, False otherwise.
        """
        return 0 <= row < 10 and 0 <= col < 10


    @staticmethod
    def get_move_directions(piece: str):
        """
        Determine possible move directions based on the type and color of a piece.

        Args:
        piece (str): Piece for which to determine possible move directions.

        Returns:
        Tuple[(int, int)]: A tuple of possible move directions.
        """
        if piece[1] == "p":
            if piece[0] == "w":
                return (-1, -1), (-1, 1)
            elif piece[0] == "b":
                return (1, -1), (1, 1)
            else:
                None

        elif piece[1] == "k":
            return (-1, -1), (-1, 1), (1, -1), (1, 1)
        else:
            return None

    @staticmethod
    def get_capture_directions():
        """
        Get all possible capture directions.

        Returns:
        Tuple[(int, int)]: A tuple of possible capture directions.
        """
        return (-1, -1), (-1, 1), (1, -1), (1, 1)

    def compute_valid_moves_and_check_terminal(self):
        """
        Compute valid moves and check if the current game state is terminal.

        Returns:
        Tuple[bool, int]: A tuple indicating whether the game is terminal and the game result.
        """
        self.compute_if_needed_and_get_single_valid_moves()
        return len(self.valid_moves) == 0, -1

    def is_terminal(self):
        """
        Check if the current game state is terminal.

        Returns:
        Tuple[bool, int]: A tuple indicating whether the game is terminal and the game result.
        """
        self.compute_if_needed_and_get_single_valid_moves()
        return len(self.valid_moves) == 0, -1

    def get_game_result_by_scoring(self):
        """
        Calculate the game result from the perspective of the current player.

        Returns:
        int: The score of the game from the perspective of the current player.
        """
        score = 0
        for row in self.board:
            for square in row:
                if square == 'wp':
                    score += 1
                elif square == 'bp':
                    score -= 1
                elif square == "wk":
                    score += 3
                elif square == "bk":
                    score -= 3
        if score > 0:
            result_from_whites_perspective = 1
        elif score < 0:
            result_from_whites_perspective = -1
        else:
            result_from_whites_perspective = 0
        return result_from_whites_perspective * (1 if self.white_to_move else -1)

    def calculate_board_score(self):
        """
        Calculate the score of the current state of the board.

        The score is calculated based on the value of each piece (white man: +1, black man: -1, white king: +3, black king: -3)

        Returns:
        int: The score of the board.
        """
        score = 0
        for row in self.board:
            for square in row:
                if square == 'wp':
                    score += 1
                elif square == 'bp':
                    score -= 1
                elif square == "wk":
                    score += 3
                elif square == "bk":
                    score -= 3
        return score

    def get_single_valid_moves(self):
        return self.single_valid_moves

    # All moves considering rules (for e., the move that captures the greatest number of pieces must be made.)
    def compute_and_get_single_valid_moves(self):
        """
        Compute the list of single valid moves, considering capturing rules.
        """
        self.single_valid_moves = []
        if self.is_capturing:
            for move_sequence in self.valid_moves:
                if move_sequence[self.capture_index] not in self.single_valid_moves:
                    self.single_valid_moves.append(move_sequence[self.capture_index])
        else:
            self.single_valid_moves = self.valid_moves

        return self.single_valid_moves

    def get_valid_moves_with_ids_for_mcts(self):
        """
        Compute the single valid moves and their corresponding IDs for Monte Carlo Tree Search (MCTS).

        Returns:
        Tuple[List[Move], List[int]]: A tuple containing a list of single valid moves and a list of their corresponding IDs.
        """
        self.single_valid_moves = []
        move_ids = []
        if self.is_capturing:
            for move_sequence in self.valid_moves:
                if move_sequence[self.capture_index] not in self.single_valid_moves:
                    self.single_valid_moves.append(move_sequence[self.capture_index])
        else:
            self.single_valid_moves = self.valid_moves

        for move in self.single_valid_moves:
            move_ids.append(move_id_mapper.MOVE_TO_ID[str(move)])

        return self.single_valid_moves, move_ids  # for now, we will not worry about some rules

    def clone(self):
        """
        Compute the single valid moves and their corresponding IDs for Monte Carlo Tree Search (MCTS).

        Returns:
        Tuple[List[Move], List[int]]: A tuple containing a list of single valid moves and a list of their corresponding IDs.
        """
        new_game = CheckersGameState()

        new_game.board = copy.deepcopy(self.board)
        new_game.white_to_move = self.white_to_move
        new_game.blocked_squares = copy.copy(self.blocked_squares)
        new_game.is_capturing = self.is_capturing
        new_game.capture_index = self.capture_index
        new_game.valid_moves = copy.copy(self.valid_moves)

        return new_game

    def get_encoded_state(self):
        """
        Get the encoded state of the checkers board.

        The encoded state is a 3D numpy array where each channel corresponds to a specific piece type (white man, black man, white king, black king),
        whether it's white's turn, blocked squares and single valid moves.

        Returns:
        np.array: The encoded state of the checkers board.
        """
        network_input = np.zeros((self.channels, 10, 10), dtype=int)
        for row in range(10):
            for col in range((row+1) % 2, 10, 2):
                if self.board[row][col] == "wp":
                    network_input[0, row, col] = 1
                    continue
                else:
                    network_input[0, row, col] = 0

                if self.board[row][col] == "bp":
                    network_input[1, row, col] = 1
                    continue
                else:
                    network_input[1, row, col] = 0

                if self.board[row][col] == "wk":
                    network_input[2, row, col] = 1
                    continue
                else:
                    network_input[2, row, col] = 0

                if self.board[row][col] == "bk":
                    network_input[3, row, col] = 1
                else:
                    network_input[3, row, col] = 0
        network_input[4] = self.white_to_move
        for row, col in self.blocked_squares:
            network_input[5, row, col] = 1

        if len(self.valid_moves) == 0:
            self.compute_if_needed_and_get_single_valid_moves()
        for move in self.single_valid_moves:
            network_input[6, move.start_row, move.start_col] = 1

        return network_input.astype(np.float32)

    def is_move_valid(self, move_str: str):
        """
        Check if a move is valid given its string representation.

        Args:
        move_str (str): The string representation of the move.

        Returns:
        bool: True if the move is valid, False otherwise.
        """
        return self.get_move_from_str(move_str) is not None

    def get_move_from_id(self, move_id: int):
        """
        Get a Move object given its ID.

        Args:
        move_id (int): The ID of the move.

        Returns:
        Move: The corresponding Move object if it's valid, None otherwise.
        """
        return self.get_move_from_str(move_id_mapper.ID_TO_MOVE[move_id])

    def get_move_from_str(self, move_str: str):
        """
        Get a Move object given its string representation.

        Args:
        move_str (str): The string representation of the move.

        Returns:
        Move: The corresponding Move object if it's valid, None otherwise.
        """
        if len(self.valid_moves) == 0:
            self.compute_if_needed_and_get_single_valid_moves()
        if 'x' in move_str:
            start_pos, end_pos = [int(pos) for pos in move_str.split('x')]
        else:
            start_pos, end_pos = [int(pos) for pos in move_str.split('-')]

        start_row = (start_pos-1)//5
        start_col = ((start_pos-1)%5)*2 + (start_row+1) % 2
        end_row = (end_pos-1)//5
        end_col = ((end_pos-1)%5)*2 + (end_row+1) % 2
        move = Move((start_row, start_col), (end_row, end_col), self.board, self.white_to_move)
        for i in range(len(self.single_valid_moves)):
            if move == self.single_valid_moves[i]:
                return self.single_valid_moves[i]
        return None


class Move:
    BOARD_DIMENSION = 10  # make this variable static constant in a python file which stores constant variables
    SQUARE_TO_ROW_COL = {}
    ROW_COL_TO_SQUARE = {}
    pos = 1
    for row in range(BOARD_DIMENSION):
        for col in range(BOARD_DIMENSION):
            if (row + col) % 2:
                SQUARE_TO_ROW_COL[pos] = (row, col)
                ROW_COL_TO_SQUARE[(row, col)] = pos
                pos += 1

    def __init__(self, start_sq, end_sq, board, is_white, captured_piece="--", captured_piece_pos=None):
        """
        Initialize a Move object.

        Args:
        start_square (tuple): The starting square coordinates (row, col).
        end_square (tuple): The ending square coordinates (row, col).
        board (list): The game board.
        is_white (bool): True if it's white's move, False otherwise.
        captured_piece (str, optional): The piece that was captured during the move. Default is "--" for no piece.
        captured_piece_position (tuple, optional): The position of the captured piece.
        """
        self.start_row = start_sq[0]
        self.start_col = start_sq[1]
        self.end_row = end_sq[0]
        self.end_col = end_sq[1]
        self.piece_moved = board[self.start_row][self.start_col]
        self.captured_piece = captured_piece
        self.captured_piece_pos = captured_piece_pos

        self.move_id = self.start_row * 1000 + self.start_col * 100 + self.end_row * 10 + self.end_col

        self.is_pawn_promotion = self.piece_moved == "wp" and self.end_row == 0 or self.piece_moved == "bp" and self.end_row == 9
        self.is_white = is_white

    def to_checkers_notation(self):
        """
        Represent the move in standard checkers notation.

        Returns:
        str: The move in standard checkers notation.
        """
        return str(self.square_position(self.start_row, self.start_col)) + \
            ("-" if self.captured_piece_pos is None else "x") + str(self.square_position(self.end_row, self.end_col))

    def square_position(self, row, col):
        """
        Get the square position corresponding to the given row and column.

        Args:
        row (int): The row of the square.
        col (int): The column of the square.

        Returns:
        int: The square position.
        """
        if (row, col) in self.ROW_COL_TO_SQUARE:
            return self.ROW_COL_TO_SQUARE[(row, col)]

    def capture_notation_extension(self):
        """
        extends a capturing move sequence in standard checkers notation.

        Returns:
        str: The second part of capturing move in standard checkers notation.
        """
        return "x" + str(self.end_row) + str(self.end_col)

    def __eq__(self, other):
        """
        Check if this move is equal to another move.

        Args:
        other (Move): The other move.

        Returns:
        bool: True if the moves are equal, False otherwise.
        """
        if isinstance(other, Move):
            return self.move_id == other.move_id

    def __str__(self):
        """
        Represent the move as a string.

        Returns:
        str: The string representation
        """
        return str(self.square_position(self.start_row, self.start_col)) + \
            ("-" if self.captured_piece_pos is None else "x") + \
            str(self.square_position(self.end_row, self.end_col))
