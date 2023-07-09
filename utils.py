import collections
import pickle
from datetime import datetime
import torch
import glob
import re
import random
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
import numpy as np
import move_id_mapper
import os
from torchinfo import summary


def load_training_data(filename):
    """
    Load training data from a file.

    Args:
    filename (str): The path to the file from which to load the training data.

    Returns:
    list: The loaded training data.
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def load_multiple_training_data(*filenames):
    """
    Load training data from multiple files.

    Args:
    filenames (str): Paths to the files from which to load the training data.

    Returns:
    list: The loaded training data from all provided files.
    """
    data = []
    for filename in filenames:
        with open(filename, 'rb') as file:
            data.extend(pickle.load(file))
    return data


def load_latest_model_and_optimizer(source_path ='data/model/', device=torch.device("cpu")):
    """
    Load the most recent AlphaZero model and its corresponding optimizer.

    Args:
    source_path (str, optional): Path to the directory that contains the model and optimizer files. Defaults to 'data/model/'.
    device (torch.device, optional): The device on which to load the model. Defaults to the CPU.

    Returns:
    tuple: The loaded AlphaZero model and optimizer.
    """
    highest_iter = get_current_model_iteration()
    return load_model_and_optimizer_by_iteration(highest_iter, source_path, device)


def load_model_and_optimizer_by_iteration(iteration_id: int, source_path ='data/model/', device=torch.device("cpu")):
    """
     Load an AlphaZero model and its corresponding optimizer based on iteration ID.

     Args:
     iteration_id (int): The iteration ID of the model and optimizer to load.
     source_path (str, optional): Path to the directory that contains the model and optimizer files. Defaults to 'data/model/'.
     device (torch.device, optional): The device on which to load the model. Defaults to the CPU.

     Returns:
     tuple: The loaded AlphaZero model and optimizer.
     """
    model_optimizer_path =  f"{source_path}model_optimizer_{iteration_id}_*pt"
    model_optimizer_path = glob.glob(model_optimizer_path)
    if len(model_optimizer_path) != 1:
        raise Exception(f"1 model_optimizer_{iteration_id} expected, but there were {len(model_optimizer_path)}")
    return load_model_and_optimizer(model_optimizer_path[0], device=device)


def load_model_and_optimizer(model_optimizer_url, device=torch.device("cpu")):
    """
    Load an AlphaZero model and its corresponding optimizer from a given URL.

    Args:
    model_optimizer_url (str): The URL of the model and optimizer file.
    device (torch.device, optional): The device on which to load the model. Defaults to the CPU.

    Returns:
    tuple: The loaded AlphaZero model and optimizer.
    """
    from pytorch_model_alphazero import AlphaZeroNet
    model = AlphaZeroNet(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    model_optimizer = torch.load(model_optimizer_url)
    model.load_state_dict(model_optimizer["model_state_dict"])
    optimizer.load_state_dict(model_optimizer["optimizer_state_dict"])
    model.name = model_optimizer_url[model_optimizer_url.rfind("/model/")+7 :]
    return model, optimizer

def generate_rotated_training_data(data_path:str):
    """
    Load training data, rotate the states and save the rotated data into a new file.

    Args:
    data_path (str): Path to the original training data file.

    Returns:
    str: The filename of the newly created file containing the rotated training data.
    """
    training_data = load_training_data(data_path)
    for index, (state, mcts_dist, actual_result) in enumerate(training_data):
        state = np.flip(state)[::-1]
        state[4] = 1 - state[4][0]
        switched_pieces_state = np.zeros(shape=state.shape, dtype=state.dtype)
        switched_pieces_state[[0, 1, 2, 3]] = state[[1, 0, 3, 2]]
        switched_pieces_state[4:7] = state[4:7]

        ids = np.where(mcts_dist > 0)[0]
        rotated_ids = [move_id_mapper.ROTATED_ID_MAPPED[id] for id in ids]
        new_mcts_dist = np.zeros(len(move_id_mapper.MOVE_TO_ID))
        new_mcts_dist[rotated_ids] = mcts_dist[ids]
        mcts_dist = new_mcts_dist

        training_data[index] = list([switched_pieces_state, mcts_dist, actual_result])

    filename = data_path.rstrip(".pkl") + "_rotated.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(training_data, file)
        print(f"{len(training_data)} game states rotated and saved under {filename}")
    return filename

def extract_capture_data(data_path:str):
    """
    Load training data, extract instances where captures are made and save them into a new file.

    Args:
    data_path (str): Path to the original training data file.

    Returns:
    str: The filename of the newly created file containing the capture data.
    """
    training_data = load_training_data(data_path)
    memory = []
    for i in range(0, len(training_data)):
        if (np.where(training_data[i][1] > 0)[0] >= 570).any():
            memory.append(training_data[i])
    filename = data_path.rstrip(".pkl") + "_captures.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(memory, file)
    print(f"{len(memory)} captures were found and saved as {filename}!")
    return filename

def merge_all_data(source="data/training_data/", dest="data/merged_training_data/", delete_files = False):
    """
    Merge all training data files from a given source directory and save the merged data into a new file.

    Args:
    source (str, optional): Path to the directory containing the training data files to merge. Defaults to "data/training_data/".
    dest (str, optional): Path to the directory where the new file with the merged data will be saved. Defaults to "data/merged_training_data/".
    delete_files (bool, optional): Whether or not to delete the original data files after merging. Defaults to False.

    Returns:
    str: The filename of the newly created file containing the merged training data.
    """
    training_data = []
    data_paths = glob.glob(source + "*.pkl")
    for data_path in data_paths:
        training_data.extend(load_training_data(data_path))

    filename = dest + 'Checkers_Merged_Data' + '_' \
               + generate_timestamp() + '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(training_data, file)
        print(f"{len(training_data)} game states from {len(data_paths)} files were merged  and saved under {filename}")

    if delete_files:
        for data_path in data_paths:
            os.remove(data_path)
        print(f"{len(data_paths)} files were deleted")
    return filename



def merge_data_files(data_fns, iteration, path='data/training_data/'):
    """
    Merge multiple training files into a single file.

    Args:
    data_fns (list): List of filenames for the training data files to merge.
    iteration (int): Iteration number to append to the filename of the merged file.
    path (str, optional): Path to the directory containing the training data files to merge. Defaults to "data/training_data/".

    Returns:
    list: The merged training data.
    """
    training_data = []
    for idx, fn in enumerate(data_fns):
        fn_dir = path + fn
        training_data.extend(load_training_data(fn_dir))
    filename = save_merged_data(training_data, iteration, generate_timestamp())
    return training_data


def save_merged_data(memory, iteration, timestamp):
    """
    Save merged training data to disk as a Pickle file.

    Args:
    memory (list): The merged training data to be saved.
    iteration (int): Iteration number to append to the filename.
    timestamp (str): Timestamp to append to the filename.

    Returns:
    str: The filename of the created file.
    """
    filename = 'data/training_data/Checkers_Data' + str(iteration) + '_' \
        + timestamp + '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(memory, file)
    return filename



def generate_timestamp():
    """
    Generate a timestamp string to be used in filenames.

    Returns:
    str: A string representation of the current date and time.
    """
    timestamp = datetime.now(tz=None)
    timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
    return timestamp_str


def get_next_model_iteration(model_path='data/model/'):
    """
    Find the highest iteration number among existing model files and increment it by one.

    Args:
    model_path (str, optional): Path to the directory containing the model files. Defaults to 'data/model/'.

    Returns:
    int: The next model iteration number.
    """
    highest_iter = -1
    for model_path in glob.glob(f"{model_path}model_optimizer_*.pt"):
        model_iter = int(re.findall('\d+|$', model_path)[0])
        if model_iter > highest_iter:
            highest_iter = model_iter
    iteration = highest_iter + 1
    return iteration

def get_current_model_iteration(model_path='data/model/'):
    """
    Find the highest iteration number among existing model files.

    Args:
    model_path (str, optional): Path to the directory containing the model files. Defaults to 'data/model/'.

    Returns:
    int: The current model iteration number.

    Raises:
    Exception: If no model files are found in the specified directory.
    """
    current_iter = -1
    for model_path in glob.glob(f"{model_path}model_optimizer_*.pt"):
        model_iter = int(re.findall('\d+|$', model_path)[0])
        if model_iter > current_iter:
            current_iter = model_iter
    if current_iter < 0:
        raise Exception("No model found!!")
    return current_iter

def log_phase_statistics(phase, **kwargs):
    """
    Record the parameters used in a phase of the training pipeline to a text file.

    Args:
    phase (str): The phase of the training pipeline. One of ['selfplay', 'training', 'evaluation', 'final'].
    **kwargs: The parameters used in the phase.

    Raises:
    ValueError: If an invalid phase is specified.
    """
    if phase == 'selfplay':
        filename = 'data/training_data/Checkers_SelfPlay_Params_' + \
                   generate_timestamp() + '.txt'
    elif phase == 'training':
        filename = 'data/training_params/Checkers_Training_Params_' + \
                   generate_timestamp() + '.txt'
    elif phase == 'evaluation':
        filename = 'data/tournament_results/Checkers_Evaluation_Params_' + \
                   generate_timestamp() + '.txt'
    elif phase == 'final':
        filename = 'data/final_eval/Checkers_Final_Evaluation_Params_' + \
                   generate_timestamp() + '.txt'
    else:
        raise ValueError('Invalid phase!')
    with open(filename, 'w') as file:
        for key, val in kwargs.items():
            file.write('{} = {}\n'.format(key, val))




def save_model_and_optimizer(model, optimizer, iteration=None, show_timestamp=True, path="data/model/"):
    """
    Save a model and its optimizer to disk.

    Args:
    model (torch.nn.Module): The model to be saved.
    optimizer (torch.optim.Optimizer): The optimizer to be saved.
    iteration (int, optional): Iteration number to append to the filename. If not provided, it's calculated from existing model files.
    show_timestamp (bool, optional): Whether to append the current timestamp to the filename. Defaults to True.
    path (str, optional): Path to the directory where the file will be saved. Defaults to "data/model/".

    Returns:
    str: The filename of the created file.
    """
    if iteration is None:
        iteration = get_next_model_iteration()

    ts = ""
    if show_timestamp:
        ts = generate_timestamp()
    model_optimizer_url = f"{path}model_optimizer_{iteration}_{ts}.pt"
    if type(model) is collections.OrderedDict:
        torch.save({"model_state_dict": model,
                    "optimizer_state_dict": optimizer}, model_optimizer_url)
    else:
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, model_optimizer_url)

    return model_optimizer_url


def split_memory(memory, train_ratio=0.8):
    """
    Shuffle and split a dataset into a training set and a validation set.

    Args:
    memory (list): The dataset to be split.
    train_ratio (float, optional): The ratio of the dataset to be used as training data. Defaults to 0.8.

    Returns:
    tuple: A tuple containing the training set and the validation set.
    """
    random.shuffle(memory)  # shuffle memory

    # Calculate split index
    split_idx = int(len(memory) * train_ratio)

    # Split memory
    train_memory = memory[:split_idx]
    val_memory = memory[split_idx:]

    return train_memory, val_memory


def save_dataset(memory, iteration, show_timestamp=True, dest_path="data/training_data/"):
    """
    Save a dataset to disk as a Pickle file.

    Args:
    memory (list): The dataset to be saved.
    iteration (int): Iteration number to append to the filename.
    show_timestamp (bool, optional): Whether to append the current timestamp to the filename. Defaults to True.
    dest_path (str, optional): Path to the directory where the file will be saved. Defaults to "data/training_data/".

    Returns:
    str: The filename of the created file.
    """
    ts = ""
    if show_timestamp:
        ts = generate_timestamp()
    filename = dest_path+'Checkers_Data' + str(iteration) + '_' \
               + ts + '_P' + '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(memory, file)
    return filename


def plot_loss_history(history, TRAINING_ITERATION):
    """
    Plot the training loss versus the training epoch and save the plot to disk.

    Args:
    history (dict): A dictionary mapping each epoch to the corresponding loss value.
    TRAINING_ITERATION (int): The iteration number to include in the plot's title and the saved file's name.

    Returns:
    str: The filename of the created plot image.
    """
    legend = list(history.keys())
    for key in history.keys():
        plt.plot(history[key])
    plt.title('Iteration {} Model Loss'.format(TRAINING_ITERATION))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper right')
    plt.grid()
    filename = 'data/plots/Checkers_Model' + str(TRAINING_ITERATION) + \
        '_TrainingLoss_' + generate_timestamp() + '.png'
    plt.draw()
    fig1 = plt.gcf()
    fig1.set_dpi(200)
    fig1.savefig(filename)
    plt.show()
    plt.close()
    return filename


def save_tourney_info(args: dict, current_model_name: str, best_model_name: str, start_time: float):
    """
    Save information about a tournament to a file.

    Args:
    args (dict): A dictionary of parameters used in the tournament.
    current_model_name (str): The name of the model used in the current iteration.
    best_model_name (str): The name of the best model found so far.
    start_time (float): The start time of the tournament, as a Unix timestamp.

    Returns:
    None
    """
    tourney_info = dict()
    tourney_info["GAME COUNT"] = args['num_selfPlay_iterations']
    tourney_info['C'] = args['C']
    tourney_info['NUMBER OF SEARCHES IN EACH MCTS'] = args['num_searches']
    tourney_info['DIRICHLET EPSILON'] = args["dirichlet_epsilon"]
    tourney_info['DIRICHLER ALPHA'] = args["dirichlet_alpha"]
    tourney_info['ELAPSED TIME'] = str(int((time.time() - start_time))) + " seconds"
    tourney_info['TIMESTAMP'] = generate_timestamp()
    tourney_info['CURRENT MODEL'] = current_model_name
    tourney_info['BEST MODEL'] = best_model_name
    log_phase_statistics('evaluation', **tourney_info)


def save_train_info(loss_history:dict, args: dict, model_from_epoch:int,
                   model_iter:int, train_memory_size:int, val_memory_size:int, new_model_name: str, start_time: float,
                    learning_rate: float, weight_decay: float):
    """
    Save information about a training phase to a file.

    Args:
    loss_history (dict): A dictionary mapping each epoch to the corresponding loss value.
    args (dict): A dictionary of parameters used in the training.
    model_from_epoch (int): The epoch from which the model was taken.
    model_iter (int): The iteration number.
    train_memory_size (int): The size of the training dataset.
    val_memory_size (int): The size of the validation dataset.
    new_model_name (str): The name of the newly trained model.
    start_time (float): The start time of the training, as a Unix timestamp.
    learning_rate (float): The learning rate used in the training.
    weight_decay (float): The weight decay used in the training.

    Returns:
    None
    """
    train_info = dict()
    train_info['TRAIN DATA'] = train_memory_size
    train_info['VALIDATION DATA'] = val_memory_size
    train_info['MAX EPOCH'] = args['num_epochs']
    train_info['PATIENCE'] = args['patience']
    train_info['BATCH SIZE'] = args['batch_size']
    train_info['MODEL FROM ITERATION'] = model_iter
    train_info['INITIAL TRAIN LOSS'] = loss_history["train_loss"][0]
    train_info['INITIAL TRAIN POLICY HEAD LOSS'] = loss_history["train_policy_loss"][0]
    train_info['INITIAL TRAIN VALUE HEAD LOSS'] = loss_history["train_value_loss"][0]
    train_info['INITIAL VALIDATION LOSS'] = loss_history["val_loss"][0]
    train_info['INITIAL VALIDATION POLICY HEAD LOSS'] = loss_history["val_policy_loss"][0]
    train_info['INITIAL VALIDATION VALUE HEAD LOSS'] = loss_history["val_value_loss"][0]
    train_info['TRAIN LOSS'] = loss_history["train_loss"][model_from_epoch]
    train_info['TRAIN POLICY HEAD LOSS'] = loss_history["train_policy_loss"][model_from_epoch]
    train_info['TRAIN VALUE HEAD LOSS'] = loss_history["train_value_loss"][model_from_epoch]
    train_info['VALIDATION LOSS'] = loss_history["val_loss"][model_from_epoch]
    train_info['VALIDATION POLICY HEAD LOSS'] = loss_history["val_policy_loss"][model_from_epoch]
    train_info['VALIDATION VALUE HEAD LOSS'] = loss_history["val_value_loss"][model_from_epoch]
    train_info["MODEL FROM EPOCH"] = model_from_epoch + 1
    train_info['NEW MODEL'] = new_model_name
    train_info['ELAPSED TIME'] = str(int((time.time() - start_time))) + " seconds"
    train_info['TIMESTAMP'] = generate_timestamp()
    train_info['LEARNING RATE'] = learning_rate
    train_info['WEIGHT DECAY'] = weight_decay
    log_phase_statistics('training', **train_info)


def save_tourney_results(game_outcomes, game_end_states=None):
    """
    Save the results of a tournament into a .txt file.

    Args:
    game_outcomes (list): A list of tuples containing the game results. Each tuple should contain game number,
        player 1's name, player 2's name, outcome, and turn count.
    game_end_states (list, optional): A list of final game states. Default is None.

    Returns:
    filename (str): The path to the saved file.
    """
    model1 = game_outcomes[0][1]
    model2 = game_outcomes[0][2]
    model1_wins, model2_wins, draws = 0, 0, 0
    for idx, outcome_list in enumerate(game_outcomes):
        outcome_list[0] = idx+1 # Renumber games
    for game_num, player1, player2, outcome, move_count in game_outcomes:
        if outcome == 1:
            if player1 == model1: model1_wins += 1
            if player1 == model2: model2_wins += 1
        elif outcome == -1:
            if player2 == model1: model1_wins += 1
            if player2 == model2: model2_wins += 1
        elif outcome == 0:
            draws += 1

    outcome_to_text = {1: "Player 1 wins", 0: "Draw", -1: "Player 2 wins"}
    game_outcomes = [[c0, c1, c2, outcome_to_text[c3], c4] for c0, c1, c2, c3, c4 in game_outcomes]


    model1_wld = str(model1_wins) + '/' + str(model2_wins) + '/' + str(draws)
    model2_wld = str(model2_wins) + '/' + str(model1_wins) + '/' + str(draws)
    summary_table = [[model1, model1_wld],[model2, model2_wld]]
    summary_headers = ['Neural Network', 'Wins/Losses/Draws']
    headers = ['Game Number', 'Player 1', 'Player 2', 'Outcome', 'Turn Count']
    filename = 'data/tournament_results/Tournament_' + generate_timestamp() + '.txt'
    with open(filename, 'w') as file:
        file.write(tabulate(summary_table, tablefmt='fancy_grid',
                            headers=summary_headers))
        file.write('\n\n')
        file.write(tabulate(game_outcomes, tablefmt='fancy_grid',
                            headers=headers))

        if game_end_states is not None:
            for game_id in range(len(game_end_states)):
                file.write(f'\n\n\n {game_id+1}. GAME END STATE ({game_outcomes[game_id][1]} VS {game_outcomes[game_id][2]}):\n')
                file.write(tabulate(game_end_states[game_id], tablefmt='fancy_grid'))

    return filename


def save_final_tourney_results(game_outcomes, game_end_states, player1, player2, game_id, device_id, validation=False):
    """
    Save the results of a final tournament into a .txt file.

    Args:
    game_outcomes (list): A list of tuples containing the game results. Each tuple should contain game number,
        player 1's name, player 2's name, outcome, turn count and terminal reason.
    game_end_states (list): A list of final game states.
    player1 (Player): A Player object for player 1.
    player2 (Player): A Player object for player 2.
    game_id (int): The ID of the game.
    device_id (int): The ID of the device.

    Returns:
    filename (str): The path to the saved file.
    """
    model1 = game_outcomes[0][1]
    model2 = game_outcomes[0][2]
    model1_wins, model2_wins, draws = 0, 0, 0
    for idx, outcome_list in enumerate(game_outcomes):
        outcome_list[0] = idx+1 # Renumber games
    for game_num, player1_name, player2_name, outcome, move_count, trm_reason in game_outcomes:
        if outcome == 1:
            if player1_name == model1: model1_wins += 1
            if player1_name == model2: model2_wins += 1
        elif outcome == -1:
            if player2_name == model1: model1_wins += 1
            if player2_name == model2: model2_wins += 1
        elif outcome == 0:
            draws += 1


    outcome_to_text = {1: "Player 1 wins", 0: "Draw", -1: "Player 2 wins"}
    game_outcomes = [[c0, c1, c2, outcome_to_text[c3], c4, c5] for c0, c1, c2, c3, c4, c5 in game_outcomes]


    model1_wld = str(model1_wins) + '/' + str(model2_wins) + '/' + str(draws)
    model2_wld = str(model2_wins) + '/' + str(model1_wins) + '/' + str(draws)
    summary_table = [[model1, model1_wld],[model2, model2_wld]]
    summary_headers = ['Neural Network', 'Wins/Losses/Draws']
    headers = ['Game Number', 'Player 1', 'Player 2', 'Outcome', 'Turn Count', 'TRM Reason']
    if not validation:
        filename = f'data/final_evaluation/'
    else:
        filename = f'data/final_evaluation_validation/'
    filename += f'{game_id}_{device_id}_{player1.name}_{player2.name}.txt'


    with open(filename, 'w') as file:
        file.write(f"game_id: {game_id}\n")
        file.write(f"numpy_random_seed: {game_id+device_id}\n")
        file.write(f"game_count: {len(game_outcomes)}\n")
        file.write(f"player1: {player1.name}\n")
        file.write(f"player2: {player2.name}\n")
        file.write(f"player1_id: {player1.player_id}\n")
        file.write(f"player2_id: {player2.player_id}\n")

        file.write(f"player1_thinking_time: {player1.thinking_time}\n")
        file.write(f"player2_thinking_time: {player2.thinking_time}\n")


        file.write(f"\nGame results from the first player's perspective:\n")
        file.write(f"wins: {model1_wins}\n")
        file.write(f"draws: {draws}\n")
        file.write(f"losses: {model2_wins}\n")

        file.write(f"\ntimestamp: {generate_timestamp()}\n")

        file.write("\n\n")

        file.write(f"**** Statistics for User ****\n")
        file.write(tabulate(summary_table, tablefmt='fancy_grid',
                            headers=summary_headers))
        file.write('\n\n')
        file.write(tabulate(game_outcomes, tablefmt='fancy_grid',
                            headers=headers))

        if game_end_states is not None:
            for game_id in range(len(game_end_states)):
                file.write(f'\n\n\n {game_id+1}. GAME END STATE ({game_outcomes[game_id][1]} VS {game_outcomes[game_id][2]}):\n')
                file.write(tabulate(game_end_states[game_id], tablefmt='fancy_grid'))

    return filename


def print_model_details():
    model, _ = load_latest_model_and_optimizer()
    print(summary(model))
