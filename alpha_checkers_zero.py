from checkers_engine import CheckersGameState
from tqdm import tqdm
from earlystopping import EarlyStopping
import utils
from modified_mcts import ModifiedMonteCarloTreeSearch, GameSession
import numpy as np
import random
import torch
import torch.nn.functional as F
import time
import torch


def phase_self_play(model, game, args) -> None:
    """
    This function performs the self-play phase of the AlphaZero algorithm.
    It simulates games and uses the outcomes to generate training data.

    :param model: Trained model to use for self-play
    :param game: Instance of the game being played
    :param args: Dictionary of parameters for self-play
    """
    model.eval()
    mcts = ModifiedMonteCarloTreeSearch(args, model)

    for iteration in tqdm(range(args['num_iterations'])):
        memory = []
        for _ in tqdm(range(args['num_selfPlay_iterations'])):
            # Save the data to avoid loss when the program is interrupted
            memory += _self_play(mcts, game, args)
        utils.save_dataset(memory, iteration)


def _self_play(mcts:ModifiedMonteCarloTreeSearch, gs: CheckersGameState(), args: dict) -> list:
    """
    This function simulates a single game of self-play, generating training data for each move.

    :param mcts: Instance of the Monte Carlo Tree Search
    :param gs: Current game state
    :param args: Dictionary of parameters for self-play
    :return: List of training examples generated during the game
    """
    memory = []
    gs.reset()

    start_time = time.time()
    counter = 0
    while True:
        action_probs = mcts.search(gs)
        memory.append([gs.get_encoded_state(), action_probs, gs.white_to_move])

        if gs.move_count < args['temperature_threshold']:
            temperature_action_probs = action_probs ** (1 / args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(gs.action_size, p=temperature_action_probs)
        else:
            action = np.argmax(action_probs)

        action = gs.get_move_from_id(action)

        gs.make_move(action)
        is_terminal, value = gs.compute_valid_moves_and_check_terminal()
        if is_terminal:
            for memory_input_id in range(len(memory)):
                memory[memory_input_id][2] = (
                    value if memory[memory_input_id][2] == gs.white_to_move else -value)
            return memory
        elif gs.move_count >= args['terminate_cnt'] and not gs.is_capturing:
            value = gs.get_game_result_by_scoring()
            for memory_input_id in range(len(memory)):
                memory[memory_input_id][2] = (
                    value if memory[memory_input_id][2] == gs.white_to_move else -value)
            return memory


def phase_train(model, optimizer, args, training_game_data, model_path="data/model/") -> None:
    """
    This function orchestrates the training phase for the AlphaGo Zero model.

    :param model: The model to be trained
    :param optimizer: The optimizer for training the model
    :param args: The training arguments, such as batch size, patience for early stopping, etc.
    :param training_game_data: The game data from self-play, which is used for training
    :param model_path: The path to save the trained model
    """
    early_stopping_criterion = EarlyStopping(patience=args['patience'], verbose=True, delta=0.003, save_multiple_models=False)
    train_memory, val_memory = utils.split_memory(training_game_data, train_ratio=0.75)

    for _ in tqdm(range(args['num_epochs'])):

        model.train()
        train_loss, train_policy_loss, train_value_loss = _train_validation(model, optimizer, train_memory, args, is_train=True, verbose=True)
        print(f"Epoch {_} - Train total loss: {train_loss}, Policy total loss: {train_policy_loss}, Value total loss: {train_value_loss}")

        model.eval()
        validation_loss, validation_policy_loss, validation_value_loss = _train_validation(model, optimizer, val_memory, args, is_train=False, verbose=True)
        print(f"Epoch {_} - Validation total loss: {validation_loss}, Valdiation Policy total loss: {validation_policy_loss}, Validation Value total loss: {validation_value_loss}")

        early_stopping_criterion(validation_loss, model, optimizer)  # check if early stopping condition is met

        if early_stopping_criterion.early_stop:
            break

    model_iter = utils.get_next_model_iteration(model_path)
    utils.save_model_and_optimizer(early_stopping_criterion.model_state_dict, early_stopping_criterion.optimizer_state_dict, model_iter, True, model_path)


def _train_validation(model, optimizer, memory, args, is_train=True, verbose=False) -> float:
    """
    This function performs either training or validation, based on the 'is_train' flag.

    :param model: The model to be trained or validated
    :param optimizer: The optimizer for training the model. It can be None if is_train is True
    :param memory: The memory used for training or validation
    :param is_train: The flag indicating whether training (True) or validation (False) is to be performed
    :return: The average loss per batch
    """
    random.shuffle(memory)
    total_loss = 0
    policy_total_loss = 0
    value_total_loss = 0
    num_batches = 0
    for batchIdx in range(0, len(memory), args['batch_size']):
        sample = memory[batchIdx: min(len(memory), batchIdx + args['batch_size'])]
        state, policy_target, value_targets = zip(*sample)

        state, policy_target, value_targets = np.array(state), np.array(policy_target), np.array(
            value_targets).reshape(-1, 1)

        state = torch.tensor(state, dtype=torch.float32, device=model.device)
        policy_targets = torch.tensor(policy_target, dtype=torch.float32, device=model.device)
        value_targets = torch.tensor(value_targets, dtype=torch.float32, device=model.device)

        out_policy, out_value = model(state)

        # multi target cross entropy
        policy_loss = F.cross_entropy(out_policy, policy_targets)
        value_loss = F.mse_loss(out_value, value_targets)
        loss = policy_loss + value_loss

        if verbose:
            print(f"Batch number: {num_batches}, Policy loss: {policy_loss.item()}"
                  f", Value loss: {value_loss.item()}, Total loss: {loss.item()}")

        total_loss += loss.item()
        policy_total_loss += policy_loss.item()
        value_total_loss += value_loss.item()
        num_batches += 1

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss / num_batches, policy_total_loss / num_batches, value_total_loss / num_batches


def phase_evaluate(current_model, best_model, game, args) -> bool:
    """
    This function conducts the evaluation phase for the AlphaGo Zero model.

    :param current_model: The currently trained model to be evaluated
    :param best_model: The best model so far
    :param game: The game for which the model is being evaluated
    :param args: The training arguments
    :return: A boolean indicating if the current model's winning percentage is greater than the set threshold
    """
    current_model.eval()
    best_model.eval()
    current_model_mcts = ModifiedMonteCarloTreeSearch(args, current_model)
    best_model_mcts = ModifiedMonteCarloTreeSearch(args, best_model)
    winning_percentage = _evaluate(current_model_mcts, best_model_mcts, game, args)
    print(f"Winning percentage of the current player: {winning_percentage}")
    return winning_percentage >= args['winning_threshold']


def _evaluate(current_model_mcts, best_model_mcts, gs, args) -> float:
    """
    This function evaluates the current model by comparing it with the best model so far.

    :param current_model_mcts: The Monte Carlo Tree Search object for the current model
    :param best_model_mcts: The Monte Carlo Tree Search object for the best model
    :param gs: The current game state
    :param args: The training arguments
    :return: The winning rate of the current model
    """
    game_outcomes = []
    game_end_states = []
    for game_num in tqdm(range(args["num_evaluation_iterations"])):
        counter = 1
        start_time = time.time()

        gs.reset()
        player1, player2 = (current_model_mcts, best_model_mcts) if game_num % 2 == 0 else (best_model_mcts, current_model_mcts)
        get_current_play_from_is_white = {True: player1, False: player2}
        while True:
            action_probs = get_current_play_from_is_white[gs.white_to_move].search(gs)
            action_id = np.argmax(action_probs)
            action = gs.get_move_from_id(action_id)

            _, value = current_model_mcts.model(torch.tensor(gs.get_encoded_state(), device=current_model_mcts.model.device).unsqueeze(0))
            print(f"Winning probability: {value.item()}")
            print(
                f"Player: {'white' if gs.white_to_move else 'black'}, move: {action}, counter: {counter}, model: {get_current_play_from_is_white[gs.white_to_move].model.name}, blocked squares: {[m for m in gs.blocked_squares]}, action: {str(action)}, current score: {gs.calculate_board_score()}")
            valid_actions, idx = gs.get_valid_moves_with_ids_for_mcts()
            print(f"Move action Probs: {[str(valid_actions[i]) + ': ' + str(action_probs[idx[i]]) for i in range(len(idx))]}")
            gs.print_board()
            counter += 1
            gs.make_move(action)
            is_terminal, value = gs.compute_valid_moves_and_check_terminal()
            if is_terminal:
                outcome = value * (1 if gs.white_to_move else -1)  # take result from white's perspective. 1 for win, -1 for lose
                break
            if gs.move_count > args['terminate_cnt']:
                value = gs.get_game_result_by_scoring()
                outcome = value * (1 if gs.white_to_move else -1)  # take result from white's perspective. 1 for win, -1 for lose
                break
        game_outcomes.append([game_num + 1, player1.model.name, player2.model.name, outcome,
                              gs.move_count])
        game_end_states.append(gs.board)
        print(f'{outcome} after {gs.move_count} moves!')

    utils.save_tourney_results(game_outcomes, game_end_states)
    game_results = np.array([outcome[3] for outcome in game_outcomes])
    print(f"Outcome: {game_results}")

    values_for_white_player = (game_results+ 1)/2
    print(f"values_for_white_player: {values_for_white_player}")

    winning_value_of_current_model = (values_for_white_player[::2]).sum() + (1-values_for_white_player[1::2]).sum()
    print()
    return winning_value_of_current_model / len(game_outcomes)



def phase_self_play_parallel(model, game, args) -> None:
    """
    Runs self-play in parallel using the provided model, game, and arguments.

    Parameters:
    model (torch.nn.Module): The model that guides the Monte Carlo Tree Search.
    game (GameState): The current state of the game.
    args (dict): Dictionary of parameters required for self-play.

    Returns:
    None
    """
    model.eval()
    mcts_parallel = ModifiedMonteCarloTreeSearch(args, model)

    for iteration in tqdm(range(args['num_iterations'])):

        memory = _self_play_parallel(mcts_parallel, game, args)
        utils.save_dataset(memory, iteration)


def _self_play_parallel(mcts_parallel: ModifiedMonteCarloTreeSearch, gs: CheckersGameState(), args:dict):
    """
    A helper function that actually executes self-play in parallel.

    Parameters:
    mcts_parallel (ModifiedMonteCarloTreeSearch): The Monte Carlo Tree Search algorithm to use during self-play.
    gs (GameState): The current state of the game.
    args (dict): Dictionary of parameters required for self-play.

    Returns:
    return_memory (list): A list of game states, action probabilities, and the player to move, representing the memory of self-play games.
    """

    start_time = time.time()
    return_memory = []
    move_count = 0
    game_sessions = [GameSession(gs) for _ in range(args['num_parallel_games'])]
    while len(game_sessions) > 0:
        move_count += 1
        print(f"counter: {move_count}, elapsed time: {time.time() - start_time}")
        mcts_parallel.parallel_search(game_sessions)
        for i in range(len(game_sessions))[::-1]:
            gr = game_sessions[i]

            root = gr.root
            gs = gr.gs
            memory = gr.memory

            action_probs = np.zeros(gs.action_size)
            valid_move_idx = [child.action_taken_id for child in root.children]
            action_probs[valid_move_idx] = [child.visit_count for child in root.children]
            action_probs /= np.sum(action_probs)

            memory.append([gs.get_encoded_state(), action_probs, gs.white_to_move])

            if move_count < args['temperature_threshold']:
                temperature_action_probs = action_probs ** (1 / args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(gs.action_size, p=temperature_action_probs)
            else:
                action = np.argmax(action_probs)
            action = gs.get_move_from_id(action)

            gs.make_move(action)
            is_terminal, value = gs.compute_valid_moves_and_check_terminal()

            if is_terminal:
                for memory_input_id in range(len(memory)):
                    memory[memory_input_id][2] = (
                        value if memory[memory_input_id][2] == gs.white_to_move else -value)

                return_memory.extend(memory)
                print(f"Game finished in {gs.get_turn_count_by_log()} moves, the remaining parallel games count: {len(game_sessions)-1}")
                gs.print_board()
                del game_sessions[i]

            elif move_count >= args['terminate_cnt'] and not gs.is_capturing:
                value = gs.get_game_result_by_scoring()
                for memory_input_id in range(len(memory)):
                    memory[memory_input_id][2] = (
                        value if memory[memory_input_id][2] == gs.white_to_move() else -value)
                return_memory.extend(memory)
                print(f"Game finished in {gs.get_turn_count_by_log()} moves, the remaining parallel games count: {len(game_sessions)-1}")
                gs.print_board()

                del game_sessions[i]

    return return_memory



sample_args = {
    "terminate_cnt": 225,
    'C': 4,
    'num_searches': 5, #  In this project, this variable was always '250' or more. Use this variable with a low value only for testing purposes
    'num_iterations': 1000,
    'num_selfPlay_iterations': 10,
    "num_evaluation_iterations": 10,
    "num_parallel_games": 10,
    'num_epochs': 100,
    'batch_size': 256,
    'temperature': 1,
    'temperature_threshold': 25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 1,
    'patience': 3,
    'winning_threshold': 0.55

}



# EXAMPLE USAGE OF DIFFERENT PHASES

## SELF-PLAY PHASE
# In this phase, the model plays against itself to generate training data.
# The generated self-play data will be saved under "data/training_data/".
# cpu performs during selfplay better
model, _ = utils.load_model_and_optimizer_by_iteration(13, device="cpu")
phase_self_play(model, CheckersGameState(), sample_args)



## PARALLEL SELF-PLAY PHASE
# This phase is similar to the self-play phase, but the games are played in parallel.
# The generated self-play data will also be saved under "data/training_data/".
# cpu performs during selfplay better
model, _ = utils.load_model_and_optimizer_by_iteration(13, device="cpu")
phase_self_play_parallel(model, CheckersGameState(), sample_args)



## TRAINING PHASE
# During the training phase, the model is trained using the generated self-play data.
# The trained model will be saved under "data/models/".
# First, merge all self-play data and generate rotated training data which ares saved under "data/taining_data/".
merged_data_path = utils.merge_all_data(delete_files=False)
derived_data_path = utils.generate_rotated_training_data(merged_data_path)
data = utils.load_multiple_training_data(merged_data_path, derived_data_path)
# Then, load the model and start training.
# For better performance, the device should be cuda for nvidia GPU and mps for Macs with ARM to use the GPU.
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model, optimizer = utils.load_model_and_optimizer_by_iteration(13, device=device)
phase_train(model, optimizer, sample_args, data)



## EVALUATION PHASE
# During the evaluation phase, the model is evaluated against the best model so far.
# The game results will be saved under "data/tournament_results/".
# First, load the models to be evaluated.
model12, _ = utils.load_model_and_optimizer_by_iteration(12)
model13, _ = utils.load_model_and_optimizer_by_iteration(13)
# Then, start the evaluation.
is_new_best_model = winning_threshold = phase_evaluate(current_model=model13, best_model=model12, game=CheckersGameState(), args=sample_args)
# Print whether the evaluated model is the new best model.
print(is_new_best_model)



