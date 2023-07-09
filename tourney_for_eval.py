
from players import *
from draw_condition_checker import Draw_Condition_Checker
import time

# Define parameters for Alpha Checkers Zero players
alphacheckers_args = {
    'C': 3,
    "terminate_cnt": 250,
    'num_searches': 400,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 1,
}

# Define player information list, each entry specifies player type and its parameters

player_information = (

    ['alphacheckers', 0, alphacheckers_args],
    ['alphacheckers', 1, alphacheckers_args],
    ['alphacheckers', 2, alphacheckers_args],
    ['alphacheckers', 3, alphacheckers_args],
    ['alphacheckers', 4, alphacheckers_args],
    ['alphacheckers', 5, alphacheckers_args],
    ['alphacheckers', 6, alphacheckers_args],
    ['alphacheckers', 7, alphacheckers_args],
    ['alphacheckers', 8, alphacheckers_args],
    ['alphacheckers', 9, alphacheckers_args],
    ['alphacheckers', 10, alphacheckers_args],
    ['alphacheckers', 11, alphacheckers_args],
    ['alphacheckers', 12, alphacheckers_args],
    ['alphacheckers', 13, alphacheckers_args],

    ['minimax', 1, False],
    ['minimax', 1, True],
    ['minimax', 2, False],
    ['minimax', 2, True],
    ['minimax', 3, False],
    ['minimax', 3, True],
    ['minimax', 4, False],
    ['minimax', 4, True],
    ['minimax', 5, False],
    ['minimax', 5, True],
    ['minimax', 6, False],
    ['minimax', 6, True],
    ['minimax', 7, False],
    ['minimax', 7, True],

    ['random'],

    ['classicalmcts', 200, 1.4141]
)


def create_player(player_info: list):
    """
    Creates a player instance based on the provided player info.

    :param player_info: A list containing the player type and its parameters.
    :return: An instance of the specified player.
    """
    if player_info[0] == 'minimax':
        player = MinimaxPlayer(player_info[1], player_info[2])
    elif player_info[0] == 'alphacheckers':
        player = AlphaCheckersZeroPlayer(player_info[1], player_info[2])
    elif player_info[0] == 'random':
        player = RandomPlayer()
    elif player_info[0] == 'classicalmcts':
        player = ClassicalMCTSPlayer(player_info[1], player_info[2])
    else:
        raise Exception("Wrong player name")
    return player


def create_players():
    """
    Creates a list of players based on the pre-defined player information.

    :return: A list of player instances.
    """
    players = []
    for player_info in player_information:
        player = create_player(player_info)
        players.append(player)
    return players


def play_game(player1: Player, player2: Player, game_id, device_id: int, gs: CheckersGameState, game_count=2, terminate_cnt=250, verbose=False, validation=False):
    """
    This function plays a game of Checkers between two provided players.

    :param player1: The first player.
    :param player2: The second player.
    :param game_id: The unique ID for the game.
    :param device_id: The ID for the device running the game.
    :param gs: The initial state of the Checkers game.
    :param game_count: The number of games to play.
    :param terminate_cnt: The maximum number of moves before termination.
    :param verbose: Flag to toggle verbose logging of the game.
    :return: None
    """
    np.random.seed(game_id+device_id)

    game_outcomes = []
    game_end_states = []

    threefold_repetition_checker = Draw_Condition_Checker()

    for game_num in range(game_count):
        white, black = (player1, player2) if game_num % 2 == 0 else (player2, player1)
        players = {True: white, False: black}
        threefold_repetition_checker.reset()
        gs.reset()
        print(f"********************** Game Number: {game_num+1} **********************")
        print(f"{white.name} VS {black.name}")

        while True:
            current_player = players[gs.white_to_move]
            start_time = time.time()
            move = current_player.get_next_move(gs)
            elapsed_time = time.time() - start_time
            current_player.thinking_time += elapsed_time
            if type(move) is not list:
                move = [move]
            for m in move:
                if verbose:
                    print(
                        f"player: {current_player.name}, move player_id: {gs.move_count}, move: {m}, color: {'white' if gs.white_to_move else 'black'}, "
                        f"total thinking time: {current_player.thinking_time}, material evaluation: {gs.calculate_board_score()}")
                    gs.print_board()
                    print("************************************************")
                gs.make_move_extended(m)
            is_terminal, value = gs.compute_valid_moves_and_check_terminal()
            if is_terminal:
                outcome_for_white = value * (1 if gs.white_to_move else -1)  # take result from white's perspective. 1 for win, -1 for lsoe
                termination_reason = "No Move"
                break
            if gs.move_count >= terminate_cnt:
                value = gs.get_game_result_by_scoring()
                outcome_for_white = value * (1 if gs.white_to_move else -1)  # take result from white's perspective. 1 for win, -1 for lsoe
                termination_reason = "Limit"
                break

            threefold_repetition_checker.add_move_state(gs)
            repetition = threefold_repetition_checker.check_threefold_repetition()
            if repetition:
                value = gs.get_game_result_by_scoring()
                outcome_for_white = value * (
                    1 if gs.white_to_move else -1)  # take result from white's perspective. 1 for win, -1 for lsoe
                termination_reason = "Repetition"
                break

        game_outcomes.append([game_num + 1, white.name, black.name, outcome_for_white,
                              gs.move_count, termination_reason])
        game_end_states.append(gs.board)
        print(f'{outcome_for_white} after {gs.move_count} moves!')
        print()

    filename = utils.save_final_tourney_results(game_outcomes, game_end_states, player1, player2, game_id, device_id, validation=validation)
    print(f"Game result of the match between {player1.name} and {player2.name} was saved under: {filename}")
    print("\n\n\n")


# Execute only if the script is the main script
if __name__ == "__main__":
    game_id = 1  # unique encounter player_id
    device_id = 0  # player_id of the device required for np.random.seed, IMac: 0, MacBook: 1000
    start_playing_from_game_id = 0  # allowing for resuming the games from where they left off
    play_until_game_id = None  # if set, stops the game playing after reaching this game_id
    for first_ai_id in range(len(player_information)-1):
        """
        Iterate through each player information in the list, skipping the last one.
        For each player, create a player instance using the information provided.
        """
        for second_ai_id in range(first_ai_id+1, len(player_information)):
            """
            For each first player, pair it with every other player that comes after it in the list.
            For each pair, if the game_id is greater than the last played game_id,
            create the second player instance and start a game between the first player and the second player.
            """
            if start_playing_from_game_id <= game_id:

                first_ai_player = create_player(player_information[first_ai_id])
                second_ai_player = create_player(player_information[second_ai_id])
                print(f"{game_id}: {first_ai_player.name} VS {second_ai_player.name}")

                play_game(first_ai_player, second_ai_player, game_id, device_id, CheckersGameState(), game_count=2, verbose=True, validation=False)
                del first_ai_player
                del second_ai_id
                last_played_game_id = game_id
            game_id += 1
            if play_until_game_id is not None and play_until_game_id <= game_id:
                break

        print("*************************")

