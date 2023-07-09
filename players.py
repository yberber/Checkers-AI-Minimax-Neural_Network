from abc import ABC, abstractmethod
import utils
from modified_mcts import ModifiedMonteCarloTreeSearch
from classical_mcts import ClassicalMCTS
import numpy as np
from checkers_engine import CheckersGameState
from checkers_minimax import find_move_minimax_alpha_beta
import time
import checkers_minimax
import torch

class Player(ABC):
    """ Abstract base class for a player in the game."""
    def __init__(self, name, player_id=None):
        self.name = name
        self.thinking_time = 0
        self.player_id = player_id
    @abstractmethod
    def get_next_move(self, gs: CheckersGameState):
        """ Abstract method to get the next move given the game state."""
        pass

    @abstractmethod
    def is_human(self):
        """ Abstract method to check if the player is a human."""
        pass

    def reset_time(self):
        self.thinking_time = 0


class HumanPlayer(Player):
    """ Represents a human player in the game."""
    def __init__(self, name="Human", player_id=None):
        """ Initializes human player with given name and player_id."""

        self.name = name
        self.id = player_id
        super().__init__(name, player_id)

    def get_next_move(self, gs: CheckersGameState):
        """ Throws not implemented exception. This function should never be called for human."""
        raise Exception("Not implemented")

    def is_human(self):
        """ Returns True indicating the player is a human."""
        return True


class MinimaxPlayer(Player):
    """ Represents a player that uses the Minimax algorithm to decide the next move.\n
    Quiescence Search is True by default"""
    def __init__(self, depth: int, quiescence_search=True, verbose=False):
        """ Initializes Minimax player with given depth and quiescence search parameters."""
        name = f"Minimax_{depth}_{'with' if quiescence_search else 'without'}_quiescence"
        player_id = 1 * (10 ** 6) + depth * (10 ** 3) + quiescence_search
        super().__init__(name, player_id)
        self.depth = depth
        self.quiescence_search = quiescence_search
        self.verbose = verbose


    def get_next_move(self, gs: CheckersGameState):
        """ Returns the next move decided by the Minimax algorithm given the game state."""
        start_time = time.time()
        moves = gs.fetch_all_possible_moves()
        checkers_minimax.counter = 0
        score = None

        if len(moves) == 1:
            next_move = moves[0]

        else:
            score, next_move = find_move_minimax_alpha_beta(gs.clone(), self.depth, -255, 255,
                                    gs.white_to_move, quiescence_search=self.quiescence_search)

        if self.verbose:
            print(f"turn: {'white' if gs.white_to_move else 'black'}, player: {self.name}, score: {score}"
                  f", thinking time: {time.time() - start_time}, "
                  f", move: {[str(m) for m in next_move] if gs.is_capturing else  str(next_move)},"
                  f" search tree node count: {checkers_minimax.counter}"
                  f", move id: {gs.move_count}")

        return next_move

    def is_human(self):
        """ Returns False indicating the player is not a human."""
        return False

class AlphaCheckersZeroPlayer(Player):
    """ Represents a player that uses the AlphaZero algorithm to decide the next move.\n
    Uses default arguments if args is None"""

    def __init__(self, model_id: int, args=None, verbose=False):
        """ Initializes AlphaZero player with given model player_id and arguments."""
        self.model = utils.load_model_and_optimizer_by_iteration(model_id)
        self.verbose = verbose
        if args is None:
            args = {
                'C': 3,
                "terminate_cnt": 250,
                'num_searches': 400,
                'dirichlet_epsilon': 0.25,
                'dirichlet_alpha': 1,
            }

        name = f"AlphaCheckersZero_{model_id}_{args['num_searches']}"
        player_id = 2 * (10 ** 6) + model_id * (10 ** 3) + args['num_searches']
        super().__init__(name, player_id)
        model, _ = utils.load_model_and_optimizer_by_iteration(model_id)
        model.eval()

        self.mcts = ModifiedMonteCarloTreeSearch(args, model)

    def get_next_move(self, gs: CheckersGameState):
        """ Returns the next move decided by the AlphaZero algorithm given the game state."""
        start_time = time.time()
        valid_single_moves = gs.compute_and_get_single_valid_moves()
        mcts_action_probs = None
        if len(valid_single_moves) == 1:
            next_move =  valid_single_moves[0]
        else:
            mcts_action_probs = self.mcts.search(gs)
            action = np.argmax(mcts_action_probs)
            next_move = gs.get_move_from_id(action)


        if self.verbose:
            _, value = self.mcts.model(torch.tensor(gs.get_encoded_state(), device=self.mcts.model.device).unsqueeze(0))
            valid_actions, idx = gs.get_valid_moves_with_ids_for_mcts()
            if mcts_action_probs is None:
                mcts_action_probs = np.zeros(gs.action_size)
                mcts_action_probs[idx[0]] = 1
            print(
                f"turn: {'white' if gs.white_to_move else 'black'}, player: {self.name}, winning prabability: {value.item()}"
                f", thinking time: {time.time() - start_time}, "
                f", move: {str(next_move)},"
                f" Move action Probs: {[str(valid_actions[i]) + ': ' + str(mcts_action_probs[idx[i]]) for i in range(len(idx))]},"
                f", move id: {gs.move_count}")

        return next_move

    def is_human(self):
        """ Returns False indicating the player is not a human."""
        return False

class RandomPlayer(Player):
    def __init__(self):
        """ Initializes Random player."""
        name = "Random_Player"
        player_id = 3 * (10 ** 6)
        super().__init__(name, player_id)

    def get_next_move(self, gs: CheckersGameState):
        """ Returns a random move given the game state."""
        moves = gs.fetch_all_possible_moves()
        idx = np.random.randint(0, len(moves))
        return moves[idx]

    def is_human(self):
        """ Returns False indicating the player is not a human."""
        return False

class ClassicalMCTSPlayer(Player):
    def __init__(self, simulation_count: int, C=1.4141):
        """ Initializes MCTS player with given simulation count."""
        name = f"Classical_MCTS_{simulation_count}"
        player_id = 4 * (10 ** 6) + simulation_count * 1
        super().__init__(name, player_id)
        self.classical_mcts = ClassicalMCTS(simulation_count, C)

    def get_next_move(self, gs):
        """ Returns the next move decided by the MCTS algorithm given the game state."""
        moves = gs.fetch_all_possible_moves()
        if len(moves) == 1:
            return moves[0]
        best_move = self.classical_mcts.search(gs)
        return best_move

    def is_human(self):
        """ Returns False indicating the player is not a human."""
        return False


