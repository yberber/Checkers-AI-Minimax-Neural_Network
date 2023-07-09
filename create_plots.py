

import re
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
import numpy as np
import warnings
import glob



pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:20,.4f}'.format)
pd.set_option('display.max_colwidth', None)
pd.set_option("display.precision", 8)
warnings.filterwarnings('ignore')


def extract_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = {}

    turn_counts_game1 = int(re.findall(r'\s\s([\d\.]+)\s', lines[30])[-1])
    turn_counts_game2 = int(re.findall(r'\s\s([\d\.]+)\s', lines[32])[-1])

    for line in lines:
        player1_id_match = re.search(r'player1_id: (\d+)', line)
        player2_id_match = re.search(r'player2_id: (\d+)', line)
        player1_match = re.search(r'player1: (.+)', line)
        player2_match = re.search(r'player2: (.+)', line)
        wins_match = re.search(r'wins: (\d+)', line)
        draws_match = re.search(r'draws: (\d+)', line)
        losses_match = re.search(r'losses: (\d+)', line)
        player1_thinking_time_match = re.search(r'player1_thinking_time: ([\d\.]+)', line)
        player2_thinking_time_match = re.search(r'player2_thinking_time: ([\d\.]+)', line)

        if player1_id_match:
            data['player1_id'] = player1_id_match.group(1)
        if player2_id_match:
            data['player2_id'] = player2_id_match.group(1)
        if player1_match:
            data['player1'] = player1_match.group(1)
        if player2_match:
            data['player2'] = player2_match.group(1)
        if wins_match:
            data['wins'] = int(wins_match.group(1))
        if draws_match:
            data['draws'] = int(draws_match.group(1))
        if losses_match:
            data['losses'] = int(losses_match.group(1))
        if player1_thinking_time_match:
            data['player1_thinking_time'] = float(player1_thinking_time_match.group(1))
        if player2_thinking_time_match:
            data['player2_thinking_time'] = float(player2_thinking_time_match.group(1))

    data['turn_counts'] = [turn_counts_game1, turn_counts_game2]
    return data


def calculate_result_dict():
    files = glob.glob("data/final_evaluation/*.txt")
    result_dict = {}
    for file in files:
        data = extract_data(file)
        if data['player1_id'] not in result_dict:
            result_dict[data['player1_id']] = [data['player1'], data['player1_thinking_time'], ((data['turn_counts'][0] + 1) // 2) + (data['turn_counts'][1] // 2),
                                        data['wins'], data['draws'], data['losses']]
        else:
            result_dict[data['player1_id']][1] += data['player1_thinking_time']
            result_dict[data['player1_id']][2] += ((data['turn_counts'][0] + 1) // 2) + (data['turn_counts'][1] // 2)
            result_dict[data['player1_id']][3] += data['wins']
            result_dict[data['player1_id']][4] += data['draws']
            result_dict[data['player1_id']][5] += data['losses']

        if data['player2_id'] not in result_dict:
            result_dict[data['player2_id']] = [data['player2'], data['player2_thinking_time'], (data['turn_counts'][0] // 2) + ((data['turn_counts'][1] + 1) // 2),
                                         data['losses'], data['draws'], data['wins']]
        else:
            result_dict[data['player2_id']][1] += data['player2_thinking_time']
            result_dict[data['player2_id']][2] += ((data['turn_counts'][0] // 2) + (data['turn_counts'][1] + 1) // 2)
            result_dict[data['player2_id']][3] += data['losses']
            result_dict[data['player2_id']][4] += data['draws']
            result_dict[data['player2_id']][5] += data['wins']

    result_dict = {k: [v[0]] + [v[1] / v[2]] + v[3:6] + [2*v[3]+v[4]]   for k, v in result_dict.items()}

    return result_dict


def plot_model_performance(df, model_name, col_name, save_path, plot_type):
    """
    This function creates and saves a bar or line plot for specified models.

    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    model_name (str): Model to be used (should be either 'AlphaCheckersZero' or 'Minimax').
    col_name (str): Column to plot ('Time per Move' or 'Score').
    save_path (str): Directory to save the plots.
    plot_type (str): Type of the plot ('bar' or 'line').

    Returns:
    None. Plots are displayed and saved as .png files.
    """

    if model_name not in ['AlphaCheckersZero', 'Minimax']:
        raise ValueError("Invalid model_name. Expected 'AlphaCheckersZero' or 'Minimax'.")

    if col_name not in ['Time per Move', 'Score']:
        raise ValueError("Invalid col_name. Expected 'Time per Move' or 'Score'.")

    # Prepare DataFrame
    model_df = df[df['Name'].str.contains(model_name)]
    model_df['Depth'] = model_df['Name'].str.extract(f'{model_name}_([0-9]+)')

    if model_name == 'Minimax':
        model_df['Quiescence'] = model_df['Name'].apply(
            lambda x: 'With Quiescence' if 'with_quiescence' in x else 'Without Quiescence')

    model_df['Depth'] = model_df['Depth'].astype(int)

    # Choose plot function based on plot type
    if plot_type == 'bar':
        plot_func = sns.barplot
    elif plot_type == 'line':
        plot_func = sns.lineplot
    else:
        raise ValueError("Invalid plot_type. Expected 'bar' or 'line'.")

    # Create and save plot
    if model_name == 'Minimax':
        if plot_type == 'bar':
            plot_func(x='Depth', y=col_name, hue='Quiescence', data=model_df)
        else:
            plot_func(x='Depth', y=col_name, hue='Quiescence', data=model_df, marker='o')

    else:
        if plot_type == 'bar':
            plot_func(x='Depth', y=col_name, data=model_df)
        else:
            plot_func(x='Depth', y=col_name, data=model_df, marker='o')


    title = f"{col_name} by {model_name} Models"
    if model_name == 'Minimax':
        title += " with and without Quiescence"

    plt.title(title)
    plt.savefig(f'{save_path}/{model_name.lower()}_{plot_type}_{col_name.replace(" ", "_").lower()}')
    plt.show()




def plot_game_results(df):
    """
    This function creates a bar plot of game results (Wins, Draws, and Losses)
    for different players, sorted by scores in descending order.

    Args:
    df (pandas.DataFrame): The dataframe containing game results. It must have the columns
    'Score', 'Wins', 'Draws', 'Losses', and 'Name'.

    Returns:
    None. A bar plot is displayed and saved as a .png file.
    """
    # Sort dataframe by score in descending order
    df_sorted = df.sort_values(by="Score", ascending=False)

    # Set figure size
    plt.figure(figsize=(15, 10))

    # Extract player names
    players = df_sorted['Name']

    # Set up x locations for bars
    x = np.arange(len(players))  # the label locations

    # Plot the bars
    plt.bar(x, df_sorted['Wins'], label='Wins')
    plt.bar(x, df_sorted['Draws'], bottom=df_sorted['Wins'], label='Draws')
    plt.bar(x, df_sorted['Losses'], bottom=df_sorted['Wins']+df_sorted['Draws'], label='Losses')

    # Set labels, title, legend, and font sizes
    plt.ylabel('Game Count', fontsize=14)
    plt.title('Counts by player and result', fontsize=16)
    plt.xticks(x, players, rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('data/final_evaluation_plots/final_win_draw_lose.png')

    # Display the plot
    plt.show()


def plot_comparison(df):
    """
    This function plots a comparison line plot for Minimax and AlphaCheckersZero models.

    Args:
    df (pandas.DataFrame): DataFrame containing the data.

    Returns:
    None. The plot is displayed.
    """

    # Prepare DataFrame for Minimax
    minimax_df = df[df['Name'].str.contains('Minimax')]
    minimax_df['Depth'] = minimax_df['Name'].str.extract('Minimax_([0-9]+)')
    minimax_df['Type'] = minimax_df['Name'].apply(lambda x: 'Minimax With Quiescence' if 'with_quiescence' in x else 'Minimax Without Quiescence')

    # Prepare DataFrame for AlphaCheckersZero
    alpha_df = df[df['Name'].str.contains('AlphaCheckersZero')]
    alpha_df['Depth'] = alpha_df['Name'].str.extract('AlphaCheckersZero_([0-9]+)')
    alpha_df['Type'] = 'AlphaCheckersZero'

    # Concatenate the DataFrames
    final_df = pd.concat([minimax_df, alpha_df])
    final_df['Depth'] = final_df['Depth'].astype(int)

    # Generate the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Depth', y='Score', hue='Type', data=final_df, marker="o")
    plt.xlabel('Depth or Iteration')
    plt.title('Score by Depth/Iteration for Minimax and AlphaCheckersZero Models')

    # Save the plot
    plt.tight_layout()
    plt.savefig('data/final_evaluation_plots/ScoreComparison_Minimax_AlphaCheckersZero.png')

    # Display the plot
    plt.show()




result_dict = calculate_result_dict()

sorted_dict = dict(sorted(result_dict.items()))

# convert to list of lists, including the key as the first item of each sublist
data_list = [value for key, value in sorted_dict.items()]
data_list = sorted_dict.values()

# column names
cols = [ 'Name', 'Time per Move', 'Wins', 'Draws', 'Losses', 'Score']

# convert to DataFrame
df = pd.DataFrame(data_list, columns=cols)
df.index += 1
print(df)




plot_model_performance(df, 'AlphaCheckersZero', 'Time per Move', 'data/final_evaluation_plots', 'bar')
plot_model_performance(df, 'AlphaCheckersZero', 'Time per Move', 'data/final_evaluation_plots', 'line')
plot_model_performance(df, 'AlphaCheckersZero', 'Score', 'data/final_evaluation_plots', 'bar')
plot_model_performance(df, 'AlphaCheckersZero', 'Score', 'data/final_evaluation_plots', 'line')

plot_model_performance(df, 'Minimax', 'Time per Move', 'data/final_evaluation_plots', 'bar')
plot_model_performance(df, 'Minimax', 'Time per Move', 'data/final_evaluation_plots', 'line')
plot_model_performance(df, 'Minimax', 'Score', 'data/final_evaluation_plots', 'bar')
plot_model_performance(df, 'Minimax', 'Score', 'data/final_evaluation_plots', 'line')

plot_game_results(df)

plot_comparison(df)



minimax_df = df[df['Name'].str.contains('Minimax')]
minimax_df['Depth'] = minimax_df['Name'].str.extract('Minimax_([0-9]+)')
minimax_df['Type'] = minimax_df['Name'].apply(lambda x: 'Minimax With Quiescence' if 'with_quiescence' in x else 'Minimax Without Quiescence')
alpha_df = df[df['Name'].str.contains('AlphaCheckersZero')]
alpha_df['Depth'] = alpha_df['Name'].str.extract('AlphaCheckersZero_([0-9]+)')
alpha_df['Type'] = 'AlphaCheckersZero'
final_df = pd.concat([minimax_df, alpha_df])
final_df['Depth'] = final_df['Depth'].astype(int)
plt.figure(figsize=(10, 6))
sns.lineplot(x='Depth', y='Score', hue='Type', data=final_df, marker="o")
plt.xlabel('Depth or Iteration')
plt.tight_layout()
plt.show()



