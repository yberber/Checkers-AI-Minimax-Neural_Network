import math

import numpy as np
import copy
import checkers_engine


class Node:
    """A node in the Monte Carlo Tree Search tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """
    def __init__(self, C, gs: checkers_engine.CheckersGameState, parent=None, action_taken=None):
        """Creates a new node with the given parameters.

        Args:
        exploration_weight (float): Exploration parameter controlling the balance between
                                    exploration and exploitation.
        game_state (CheckersGameState): The state of the game at this node.
        parent (TreeNode, optional): The parent of this node.
        action_taken (str, optional): The action taken to reach this node.
        """
        self.gs = gs
        self.C = C
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_moves = copy.copy(gs.fetch_valid_moves())
        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        """Checks if the node is fully expanded.

        Returns:
        bool: True if the node is fully expanded, False otherwise.
        """
        return len(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        """Selects a child of the node.

        Returns:
        Node: The selected child node.
        """
        best_child = None
        best_ucb = -np.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        """Calculates the Upper Confidence Bound (UCB) of a child node.

        Args:
        child (Node): The child node.

        Returns:
        float: The UCB of the child node.
        """
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.C * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def expand(self):
        """Expands a node by creating a new child node.

        Returns:
        Node: The created child node.
        """
        action_id = np.random.randint(0, len(self.expandable_moves))
        action = self.expandable_moves[action_id]
        self.expandable_moves.pop(action_id)
        child_state = self.gs.clone()
        child_state.make_move_extended(action)
        child = Node(self.C, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        """Performs a random simulation/rollout to get the value at the current node.

        Returns:
        float: The value at the current node.
        """
        is_terminal, value = self.gs.compute_valid_moves_and_check_terminal()

        if is_terminal:
            return value
        rollout_state = self.gs.clone()
        while True:
            valid_moves = rollout_state.fetch_all_possible_moves()
            action_id = np.random.randint(0, len(valid_moves))
            action_sequence = valid_moves[action_id]
            rollout_state.make_minimax_move(action_sequence)
            is_terminal, value = rollout_state.compute_valid_moves_and_check_terminal()
            if is_terminal:
                return (value if self.gs.white_to_move == rollout_state.white_to_move else 1-value)


    def backpropagate(self, value):
        """Updates the node and its ancestors with the simulated value.

          Args:
          value (float): The simulated value of the current node.
          """
        self.value_sum += value
        self.visit_count += 1
        value = -value
        if self.parent is not None:
            self.parent.backpropagate(value)

class ClassicalMCTS:
    """A class representing the Monte Carlo Tree Search algorithm."""

    def __init__(self, simulation_count: int, C: float):
        """Initializes the MCTS.

        Args:
        simulation_count (int): The number of simulations to run for each MCTS.
        exploration_weight (float): The exploration weight in the UCB calculation.
        """
        self.simulation_count = simulation_count
        self.C = C

    def search(self, gs: checkers_engine.CheckersGameState) -> str:
        """Runs the MCTS algorithm and returns the best action.

        Args:
        game_state (CheckersGameState): The current state of the game.

        Returns:
        Move: The best action
        """
        # define root
        root = Node(self.C, gs)

        for search in range(self.simulation_count):
            # print(f"Search player_id: {search}")
            node = root

            while node.is_fully_expanded():
                node = node.select()
            is_terminal, value = node.gs.compute_valid_moves_and_check_terminal()    # value = 1 if white won, value = 0 if black won

            if not is_terminal:
                node = node.expand()
                value = node.simulate()
            node.backpropagate(value)
        most_visit_cnt = -1
        best_child = None
        for child in root.children:
            # quality = 1 - ((child.value_sum / child.visit_count) + 1) / 2
            # print(f"move: {child.action_taken}, visit_count: {child.visit_count}, quality: {quality}, reward: {child.value_sum*-1}")
            if most_visit_cnt < child.visit_count:
                most_visit_cnt = child.visit_count
                best_child = child
        return best_child.action_taken

