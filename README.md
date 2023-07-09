# Checkers-AI-Minimax-Neural_Network

International Draughts is very popular in many countries, yet it remains an unsolved board game due to its high state complexity (number of distinct states) with an upper bound of  10^30 game tree complexity (number of different possible sequences of legal moves) estimated at 10^54 [14]. This work aims to develop two distinct agents for playing checkers using two fundamentally different AI approaches.

The first approach employs the Minimax Algorithm, customized for checkers, and enhances its performance using alpha-beta pruning and quiescence search.

The second approach employs a trained neural network. Due to the scarcity of high-quality international checkers game data available online, the AlphaZero methodology from Google's DeepMind was adopted and trained from scratch for this purpose.

The goal was to develop two agents playing at a reasonably high skill level using completely different techniques and comparing their advantages and disadvantages, with a focus on comparing their respective advantages and disadvantages.

This project, fully implemented in Python, met all the proposed goals.


## Setup

Python 3.10 or newer is required.

Here's how to set up the project:

### Option 1: Regular Python Environment

1. Clone the repository:
```
git clone https://github.com/yourusername/alpha-checkers.git
cd alpha-checkers
```

2. (Optional but recommended) Create a new virtual environment:
```
python -m venv venv
source venv/bin/activate  # Unix
.\venv\Scripts\activate   # Windows
```

3. Install the dependencies using pip:
```
pip install -r requirements.txt
```

### Option 2: Conda Environment

1. Clone the repository:
```
git clone https://github.com/yourusername/alpha-checkers.git
cd alpha-checkers
```

2. Create a new conda environment and activate it:
```
conda create -n alpha-checkers-env python=3.10
conda activate alpha-checkers-env
```

3. Install the dependencies using pip:
```
pip install -r requirements.txt
```



## Playing the Game

To start the game with a user interface, run:

```shell
python checkers_main.py
```

This will start the game. Players may be human players or any developed agent. You can select your desired options in the user interface. You can modify the checkers_main.py script. 


## Developing and Evaluating Alpha-Checkers-Zero

The file `alpha_checkers_zero.py` is used for the development cycle of the AlphaZero-like model.

Before running the script, please command command out the call of different phases. The script contains sample starting of different phases

```shell
python alpha_checkers_zero.py
```


### Evaluation Data

The directory `data/final_evaluation` contains game results for matches played between agents, which was used to evaluate their performance. The directory `data/final_evaluation_plots` is used to save plots created using the data under `data/final_evaluation`.

Enjoy the game!

---
