"""
This is out main driver file/ It will be responsible for handling
user input and displaying the current GameState object.
"""
import math
import pygame
from checkers_engine import *
from players import *
from draw_condition_checker import Draw_Condition_Checker


BOARD_WIDTH = BOARD_HEIGHT = 640
INFO_PANEL_WIDTH = MOVE_LOG_PANEL_WIDTH = 260
MOVE_LOG_PANEL_HEIGHT = int(BOARD_HEIGHT * 0.8)
INFO_PANEL_HEIGHT = int(BOARD_HEIGHT * 0.2)
DIMENSION = 10  # dimensions of a checkers board are 10x10
SQ_SIZE = BOARD_HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}



# The main driver for our code. This will handle user input and update the graphics
def main(white, black):
    """
    Main function to handle game logic, user interaction, and updates the graphics.
    It's the entry point for the game execution.

    Parameters:
    white (Player): The white player, can be a human or AI player.
    black (Player): The black player, can be a human or AI player.

    The function initializes the pygame environment, loads the images, and enters into a game loop.
    During each iteration of the game loop, it processes pygame events, performs actions based on user
    input, makes moves for AI players, updates the game state, and refreshes the display.
    It also handles special conditions like threefold repetition and end of the game.
    """
    pygame.init()
    screen = pygame.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
    clock = pygame.time.Clock()
    screen.fill((255, 255, 255))
    move_log_font = pygame.font.SysFont("Arial", 12, False, False)
    gs = CheckersGameState()
    threefold_repetition_checker = Draw_Condition_Checker(undo_possible=True)
    valid_moves = gs.compute_if_needed_and_get_single_valid_moves()
    move_made = False  # flag variable for when a move is made
    load_images()  # only do this once, before the while loop
    running = True
    sq_selected = ()  # no square is selected, keep track of the last click of the used (tuple: (row, col))
    player_clicks = []  # keep track of player clicks (two tuples: [(6, 4), (5, 3)])
    possible_moves_for_selected = []
    game_over = False
    paused = True  # can be used to pause the game while playing AI. User can press enter to pause the game
    squares_to_highlight = []
    blurred_pieces_with_pos = []

    players = {True: white, False: black}

    while running:
        current_player = players[gs.white_to_move]
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            # mouse handler
            elif e.type == pygame.MOUSEBUTTONDOWN and current_player.is_human() and not game_over:
                location = pygame.mouse.get_pos()  # (x, y) location of mouse
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE
                if sq_selected == (row, col) or col >= 10:  # the user clicked the same square or user clicked mouse log
                    sq_selected =  ()  # deselect
                    player_clicks = []  # clear player clicks
                    possible_moves_for_selected = []
                else:
                    sq_selected = (row, col)
                    player_clicks.append(sq_selected)  # append for both 1st and 2nd clicks
                if len(player_clicks) == 1:
                    possible_moves_for_selected = gs.fetch_valid_moves_for_piece(sq_selected)
                if len(player_clicks) == 2:  # after 2nd click
                    move = Move(player_clicks[0], player_clicks[1], gs.board, gs.white_to_move)
                    possible_moves_for_selected = []

                    for i in range(len(valid_moves)):
                        if move == valid_moves[i]:
                            gs.make_move(valid_moves[i])
                            move_made = True
                            sq_selected = ()  # reset user clicks
                            player_clicks = []
                            valid_moves[i]
                            sq_selected = (0,0)
                    if not move_made:
                        player_clicks = [sq_selected]
                        possible_moves_for_selected = gs.fetch_valid_moves_for_piece(sq_selected)

            # key handle
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_z:  # undo when 'z' is pressed
                    gs.undo_move()
                    squares_to_highlight, blurred_pieces_with_pos = get_highlighting_for_captures(gs)
                    threefold_repetition_checker.update_move_entries_after_undo(gs)
                    valid_moves = gs.compute_if_needed_and_get_single_valid_moves()
                    game_over = False
                if e.key == pygame.K_SPACE:
                    paused = not paused
                if e.key == pygame.K_r:  # reset the board when 'r' is pressed
                    gs = CheckersGameState()
                    player_clicks = []
                    sq_selected = ()
                    valid_moves = gs.compute_if_needed_and_get_single_valid_moves()
                    possible_moves_for_selected = []
                    game_over = False
                    move_made = False

        # AI Move Finder Logic
        if not current_player.is_human() and not game_over and not paused:
            ai_move = current_player.get_next_move(gs)

            if ai_move is None:
                raise ValueError("Output of Minimax can not be None")

            if type(ai_move) is list:
                for index in range(0, len(ai_move)-1):
                    if ai_move[index] in valid_moves:
                        gs.make_move(ai_move[index])
                        animate_move(gs.move_log[-1], screen, gs.board, clock)
                        draw_game_state(screen, gs, possible_moves_for_selected, sq_selected, move_log_font, squares_to_highlight, blurred_pieces_with_pos, white, black)
                        pygame.display.flip()
                        valid_moves = gs.compute_if_needed_and_get_single_valid_moves()
                    else:
                        raise ValueError(
                            f"Selected move is {ai_move}, but valid moves are {[str(m) for m in valid_moves]}")
                if ai_move[-1] in valid_moves:
                    gs.make_move(ai_move[-1])
                else:
                    raise ValueError(f"Selected move is {ai_move[-1]}, but valid moves are {[str(m) for m in valid_moves]}")

            else:
                if ai_move in valid_moves:
                    gs.make_move(ai_move)
                else:
                    raise ValueError(f"Selected move is {ai_move}, but valid moves are {[str(m) for m in valid_moves]}")
            move_made = True
        if move_made:
            animate_move(gs.move_log[-1], screen, gs.board, clock, blurred_pieces_with_pos)
            valid_moves = gs.compute_if_needed_and_get_single_valid_moves()
            squares_to_highlight, blurred_pieces_with_pos = get_highlighting_for_captures(gs)
            threefold_repetition_checker.add_move_state(gs)

            if current_player.is_human() and gs.capture_index > 0:
                sq_selected = (gs.move_log[-1].end_row, gs.move_log[-1].end_col)
                player_clicks = [sq_selected]
                possible_moves_for_selected = gs.fetch_valid_moves_for_piece(sq_selected)
            move_made = False
            if len(valid_moves) == 0:
                game_over = True
                if gs.white_to_move:
                    draw_end_game_text(screen, "Black Wins")
                else:
                    draw_end_game_text(screen, "White Wins")
            elif threefold_repetition_checker.check_threefold_repetition():
                game_over = True
                draw_end_game_text(screen, "Draw")

        if not game_over:
            draw_game_state(screen, gs, possible_moves_for_selected, sq_selected, move_log_font, squares_to_highlight, blurred_pieces_with_pos, white, black)
        clock.tick(MAX_FPS)
        pygame.display.flip()


def load_images():
    """
    Load all the piece images from the images directory and resizes them to the square size.
    Also creates blurred version of images with reduced opacity.

    This function modifies global IMAGES dictionary in place, assigning to each piece type
    a corresponding Pygame surface.
    """
    pieces = ["wp", "wk", "bp", "bk"]
    for piece in pieces:
        image = pygame.image.load("images/" + piece + ".png")
        IMAGES[piece] = pygame.transform.scale(image, (SQ_SIZE, (image.get_height() / image.get_width()) * SQ_SIZE))
        IMAGES[piece + str("_c")] = IMAGES[piece].copy()
        IMAGES[piece + str("_c")].set_alpha(100)  # blurred images

# Responsible for all the graphics within a current game state.
def draw_game_state(screen, gs, possible_moves, sq_selected, log_font, squares_to_highlight, blurred_pieces_with_pos, white, black):
    """
    Responsible for all the graphics within a current game state.

    Parameters:
    screen (pygame.Surface): The surface to draw on.
    gs (GameState): The current game state.
    possible_moves (list): The list of possible moves.
    sq_selected (tuple): The selected square coordinates (row, column).
    log_font (pygame.Font): The font for rendering log texts.
    squares_to_highlight (list): List of squares (coordinates as tuples) to highlight.
    blurred_pieces_with_pos (list): List of blurred pieces with their positions.
    white (Player): The player plays with white pieces
    black (Player): The player plays with black pieces
    """
    draw_board(screen)  # draw squares on the board
    highlight_squares(screen, gs, possible_moves, sq_selected)
    highlight_last_move(screen, squares_to_highlight, blurred_pieces_with_pos)
    draw_pieces(screen, gs.board)  # draw pieces on top of those squares
    draw_move_log(screen, gs, log_font)
    draw_info_log(screen, log_font, white, black )


def highlight_last_move(screen, squares_to_highlight, blurred_pieces_with_pos):
    """
    Highlights the last move made and places a blurred image on captured pieces.

    Parameters:
    screen (pygame.Surface): The surface to draw on.
    squares_to_highlight (list): List of squares (coordinates as tuples) to highlight.
    blurred_pieces_with_pos (list): List of blurred pieces with their positions.
    """
    surface = pygame.Surface((SQ_SIZE, SQ_SIZE))
    surface.set_alpha(100)  # transparency value -> 0 0 transparent; 255 opaque
    surface.fill(pygame.Color("lightgreen"))
    for square in squares_to_highlight:
        screen.blit(surface, (square[1] * SQ_SIZE, square[0] * SQ_SIZE))
    for piece, pos in blurred_pieces_with_pos:
        screen.blit(IMAGES[piece],
                    pygame.Rect(pos[1] * SQ_SIZE, pos[0] * SQ_SIZE + 10,
                                SQ_SIZE, SQ_SIZE))


def get_highlighting_for_captures(gs:CheckersGameState):
    """
    Highlights the start and end squares of a capturing move and blurs the captured piece.
    Only captures made by the current player are considered.

    Parameters:
    gs (CheckersGameState): The current game state.

    Returns:
    squares_to_highlight (list): List of squares (coordinates as tuples) to highlight.
    blurred_pieces_with_pos (list): List of blurred pieces with their positions.
    """
    if len(gs.move_log):
        last_move_is_white = gs.move_log[-1].is_white
        squares_to_highlight = []
        blurred_pieces_with_pos = []
        last_move = gs.move_log[-1]
        show_captures_pieces = gs.white_to_move == last_move.is_white
        squares_to_highlight.append((last_move.end_row, last_move.end_col))
        for previous_move_id in range(1, len(gs.move_log) + 1):
            move = gs.move_log[-previous_move_id]

            if last_move_is_white != move.is_white:
                break
            if show_captures_pieces and move.captured_piece != "--":
                blurred_pieces_with_pos.append((move.captured_piece + str("_c"), move.captured_piece_pos))
            squares_to_highlight.append((move.start_row, move.start_col))
        return squares_to_highlight, blurred_pieces_with_pos


# Draw the squares on the board.
def draw_board(screen):
    """
    Draw the squares on the board.

    Parameters:
    screen (pygame.Surface): The surface to draw on.

    This function uses the global variable 'colors' which should be a list
    of two pygame.Color objects to represent the colors of the squares. The
    function also labels each square with a unique id, displayed on the board.
    """
    global colors
    colors = [pygame.Color("white"), pygame.Color("gray")]

    square_id = 1
    font = pygame.font.SysFont("Arial", 11, True, False)

    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color_id = (row + col) % 2
            color = colors[color_id]
            pygame.draw.rect(screen, color, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            if color_id == 1:
                text_obj = font.render(str(square_id), True, pygame.Color("black"))
                square_id += 1
                screen.blit(text_obj, (col * SQ_SIZE, row * SQ_SIZE))


# Highlight square selected and moves for piece selected
def highlight_squares(screen, gs, possible_moves, sq_selected):
    """
    Highlight square selected and moves for piece selected.

    Parameters:
    screen (pygame.Surface): The surface to draw on.
    gs (GameState): The current game state.
    possible_moves (list): The list of possible moves.
    sq_selected (tuple): The selected square coordinates (row, column).

    If the selected square is a piece that can be moved, the square will be
    highlighted, as will all possible moves for that piece.
    """
    if sq_selected != ():
        row, col = sq_selected
        if gs.board[row][col][0] == ("w" if gs.white_to_move else "b"):  # sq_selected is a piece that can be moved
            surface = pygame.Surface((SQ_SIZE, SQ_SIZE))
            surface.set_alpha(100)  # transparency value -> 0 0 transparent; 255 opaque
            surface.fill(pygame.Color("darkgreen"))
            screen.blit(surface, (col * SQ_SIZE, row * SQ_SIZE))
            surface = pygame.Surface((SQ_SIZE, SQ_SIZE))
            surface.set_alpha(100)  # transparency value -> 0 0 transparent; 255 opaque
            for move in possible_moves:
                pygame.draw.circle(surface, pygame.Color("darkgreen"), (SQ_SIZE//2,
                                                                       SQ_SIZE//2), SQ_SIZE/8)
                screen.blit(surface, (move.end_col * SQ_SIZE, move.end_row * SQ_SIZE))


# Draw the pieces on the board using the current GameState.board
def draw_pieces(screen, board, blurred_pieces_with_pos=[]):
    """
    Draw the pieces on the board using the current GameState.board.

    Parameters:
    screen (pygame.Surface): The surface to draw on.
    board (list): 2D list representing the current state of the game.
    blurred_pieces_with_pos (list, optional): List of blurred pieces with their positions.

    This function uses the global dictionary 'IMAGES' which should map piece
    types to their corresponding Pygame surfaces.
    """
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = board[row][col]
            if piece != "--":  # not empty square
                screen.blit(IMAGES[piece], pygame.Rect(col * SQ_SIZE, row * SQ_SIZE + 10, SQ_SIZE, SQ_SIZE))
    for piece, pos in blurred_pieces_with_pos:
        screen.blit(IMAGES[piece],
                    pygame.Rect(pos[1] * SQ_SIZE, pos[0] * SQ_SIZE + 10,
                                SQ_SIZE, SQ_SIZE))

def draw_info_log(screen, font, white_player, black_player):
    """
    Draw a log of useful information on the screen.

    Parameters:
    screen (pygame.Surface): The surface to draw on.
    font (pygame.Font): The font for rendering log texts.
    white_player (Player): The player object representing the white player.
    black_player (Player): The player object representing the black player.

    The function displays the names of the white and black players, and
    controls for the game (undo, reset, start/stop AI).
    """
    move_log_rect = pygame.Rect(BOARD_WIDTH, MOVE_LOG_PANEL_HEIGHT, INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, pygame.Color("black"), move_log_rect)
    info_text = []
    info_text.append("----------------------------------------------")
    info_text.append(f"white: {white_player.name}")
    info_text.append(f"black: {black_player.name}")
    info_text.append(f"'Z': undo")
    info_text.append(f"'R': Reset")
    info_text.append(f"'Space': Start/Stop AI")
    padding_x = padding_y = 5
    y_pos = padding_y
    new_line_increment_y_by = 20
    for i in range(len(info_text)):
        log_text = font.render(info_text[i], True, pygame.Color("white"))
        screen.blit(log_text, (BOARD_WIDTH + padding_x, y_pos + MOVE_LOG_PANEL_HEIGHT))
        y_pos += new_line_increment_y_by

# Draw the move log
def draw_move_log(screen, gs, font):
    """
    Draw the move log on the screen.

    Parameters:
    screen (pygame.Surface): The surface to draw on.
    gs (GameState): The current game state.
    font (pygame.Font): The font for rendering log texts.

    The function creates a visual log of all moves made in the game so far,
    formatted in the checkers notation.
    """
    move_log_rect = pygame.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
    pygame.draw.rect(screen, pygame.Color("black"), move_log_rect)

    move_log = gs.move_log
    move_texts = []

    move_sequence_number = 1
    is_white_move = True
    move_string = ""
    for i in range(len(move_log)):
        move = move_log[i]
        if is_white_move == move.is_white and is_white_move:
            move_texts.append(move_string)
            move_string = str(move_sequence_number)
            if move_sequence_number < 10:
                move_string += "  "
            move_sequence_number += 1

        move = move_log[i]

        if is_white_move == move.is_white:
            move_string += "     " + move.to_checkers_notation()
        else:
            move_string += move.capture_notation_extension()
        is_white_move = not move.is_white
    if move_string != "":
        move_texts.append(move_string)
    padding_x = padding_y = 5
    y_pos = padding_y
    new_line_increment_y_by = 20

    adjust_text_starting_index = max(1, (len(move_texts) - ((MOVE_LOG_PANEL_HEIGHT - padding_y) //  new_line_increment_y_by  )))

    for i in range(adjust_text_starting_index, len(move_texts)):
        log_text = font.render(move_texts[i], True, pygame.Color("white"))
        screen.blit(log_text, (BOARD_WIDTH + padding_x, y_pos))
        y_pos += new_line_increment_y_by


# animating a move
def animate_move(move, screen, board, clock, blurred_pieces_with_pos=[]):
    """
    Animate a move on the screen.

    Parameters:
    move (Move): The move to be animated.
    screen (pygame.Surface): The surface to draw on.
    board (list): 2D list representing the current state of the game.
    clock (pygame.time.Clock): The clock object to control the frame rate.
    blurred_pieces_with_pos (list, optional): List of blurred pieces with their positions.

    The function creates a smooth animation of a piece moving from its
    original position to its destination.
    """
    global colors
    d_r = move.end_row - move.start_row
    d_c = move.end_col - move.start_col
    frames_per_square = 4  # frames to move one square
    frame_count = int(math.sqrt(d_r*d_r + d_c*d_c) * frames_per_square)
    for frame in range(frame_count + 1):
        d_frame = frame/frame_count
        row, col = (move.start_row + d_r * d_frame, move.start_col + d_c * d_frame)
        draw_board(screen)
        draw_pieces(screen, board, blurred_pieces_with_pos)


        # erase the piece moved from its ending square
        color = colors[(move.end_row + move.end_col) % 2]
        end_square = pygame.Rect(move.end_col*SQ_SIZE, move.end_row*SQ_SIZE, SQ_SIZE, SQ_SIZE)
        pygame.draw.rect(screen, color, end_square)
        if move.captured_piece != "--":
            captured_row, captured_col = move.captured_piece_pos
            screen.blit(IMAGES[move.captured_piece], [captured_col * SQ_SIZE, captured_row * SQ_SIZE + 10, SQ_SIZE, SQ_SIZE])
        # draw moving piece
        screen.blit(IMAGES[move.piece_moved], pygame.Rect(col * SQ_SIZE, row * SQ_SIZE + 10, SQ_SIZE, SQ_SIZE))
        pygame.display.flip()
        pygame.display.update()
        clock.tick(30)

def draw_end_game_text(screen, text):
    """
    Draw an end game text on the screen.

    Parameters:
    screen (pygame.Surface): The surface to draw on.
    text (str): The text to display.

    The function displays a message, typically indicating the end of the game,
    at the center of the screen.
    """
    font = pygame.font.SysFont("Arial", 32, True, False)
    text_obj = font.render(text, True, pygame.Color("Gray"))
    text_location = pygame.Rect((BOARD_WIDTH - text_obj.get_width()) // 2, (BOARD_HEIGHT - text_obj.get_height()) // 2,
                                text_obj.get_width(), text_obj.get_height())
    screen.blit(text_obj, text_location)
    text_obj = font.render(text, True, pygame.Color("Black"))
    screen.blit(text_obj, text_location.move(2, 2))
    pygame.display.update()


if __name__ == '__main__':
    """
    Main execution block for when this file is run directly.

    It creates two AI players, white and black
    and starts the game.
    """
    minimax_player = MinimaxPlayer(depth=7, quiescence_search=True, verbose=True)
    alpha_checkers_player = AlphaCheckersZeroPlayer(model_id=13, verbose=True)
    human_player = HumanPlayer("Human")

    white = human_player
    black = alpha_checkers_player

    main(white, black)


