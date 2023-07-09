import math
import numpy as np
import torch
from checkers_engine import CheckersGameState
import move_id_mapper


class Node:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) tree.
    """
    def __init__(self, args, gs: CheckersGameState, parent=None, action_taken_id=None, prior=0, visit_count=0):
        """
        Initializes a new node with the given parameters.

        :param args: Arguments needed for the node.
        :param gs: The game state associated with the node.
        :param parent: The parent node. Default is None for root nodes.
        :param action_taken_id: The action taken to reach this node from its parent node.
        :param prior: The prior probability of this node (from parent's perspective).
        :param visit_count: The number of times this node has been visited during MCTS.
        """
        self.gs = gs
        self.args = args
        self.parent = parent
        self.action_taken_id = action_taken_id
        self.prior = prior
        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0

    def is_leaf_node(self):
        """
        Checks if the node is a leaf node (i.e., has no children)

        :return: True if the node is a leaf node, False otherwise.
        """
        return len(self.children) == 0

    def get_child_puct(self, child):
        """
        Calculates and returns the PUCT value (a measure of how promising this child node is for exploration) for a given child node.

        :param child: A child node of the current node.
        :return: The calculated PUCT value for the child node.
        """
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = child.value_sum / child.visit_count
            if self.gs.white_to_move != child.gs.white_to_move:
                q_value = - q_value
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior



class GameSession:
    """
    Class to store the state of a game and associated memory, along with a root node for MCTS and a player player_id.

    Attributes:
    gs (CheckersGameState): The game state associated with this game session.
    memory (list): A list to store specific information throughout the game.
    root_node (MCTSNode): The root node of the MCTS tree.
    current_node (MCTSNode): The current active node in the MCTS tree.
    current_player_id (int): The ID of the player whose turn it is currently.
    """
    def __init__(self, game, player_id=None):
        self.gs = game.clone()
        self.memory = []
        self.root = None
        self.node = None
        self.player_id = player_id


class ModifiedMonteCarloTreeSearch:
    """
    Represents the modified version of the Monte Carlo Tree Search (MCTS) for AlphaZero approach.
    """
    def __init__(self, args: dict, model):
        """
        Initializes the MCTS with given arguments and model.

        :param args: A dictionary of arguments for MCTS.
        :param model: The model to be used for value and policy predictions.
        """
        self.args = args
        self.model = model

    def expand(self, node, policy, valid_moves):
        """
        Expands the given node by adding children nodes for each valid move.

        :param node: The node to be expanded.
        :param policy: The policy distribution over the actions.
        :param valid_moves: List of valid moves from the current node.
        """
        for action in valid_moves:
            child_state = node.gs.clone()
            child_state.make_move(action)
            action_id = move_id_mapper.MOVE_TO_ID[str(action)]
            child = Node(self.args, child_state, node, action_id, policy[action_id])
            node.children.append(child)

    def backpropagate(self, node, value, is_white_player_terminated):
        """
        Backpropagates the value of a leaf node back up to its ancestors.

        :param node: The node from which the backpropagation begins.
        :param value: The value to be backpropagated.
        :param is_white_player_terminated: Boolean flag indicating whether the white player is terminated.
        """
        node.value_sum += (value if node.gs.white_to_move == is_white_player_terminated else -value)
        node.visit_count += 1
        if node.parent is not None:
            self.backpropagate(node.parent, value, is_white_player_terminated)

    def select_leaf_node(self, node: Node):
        """
        Selects the leaf node from the current node for expansion.

        :param node: The node from which the leaf node selection starts.
        :return: The selected leaf node.
        """
        if node.is_leaf_node():
            return node
        best_child = None
        best_puct = -np.inf
        for child in node.children:
            puct = node.get_child_puct(child)
            if puct > best_puct:
                best_child = child
                best_puct = puct
        return self.select_leaf_node(best_child)

    def apply_valid_move_mask_to_policy(self, policy, valid_moves, add_noise=False):
        """
        Applies a mask to the policy to invalidate the illegal moves and optionally add noise to the policy.

        :param policy: The policy distribution over the actions.
        :param valid_moves: List of valid moves from the current node.
        :param add_noise: Boolean flag indicating whether to add Dirichlet noise to the policy.
        """
        mask_valid_moves = np.zeros(978)
        valid_move_idx = [move_id_mapper.MOVE_TO_ID[str(move)] for move in valid_moves]
        mask_valid_moves[valid_move_idx] = 1
        policy *= mask_valid_moves
        policy /= np.sum(policy)
        if add_noise:
            dirichlet_noise = np.random.dirichlet([self.args['dirichlet_alpha']] * len(valid_moves))
            policy[valid_move_idx] = (1 - self.args['dirichlet_epsilon']) * policy[valid_move_idx] + self.args['dirichlet_epsilon'] * dirichlet_noise

    @torch.no_grad()
    def search(self, gs: CheckersGameState):
        """
        Performs the MCTS search from the given game state and returns the action probabilities.

        :param gs: The current game state.
        :return: A list of action probabilities for each action.
        """
        root = Node(self.args, gs)
        policy, _ = self.model(torch.tensor(gs.get_encoded_state(), device=self.model.device).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        valid_moves = gs.compute_if_needed_and_get_single_valid_moves()
        self.apply_valid_move_mask_to_policy(policy, valid_moves, True)
        self.expand(root, policy, valid_moves)
        self.backpropagate(root, 0, True)

        for search in range(self.args['num_searches']):
            node = self.select_leaf_node(root)
            is_terminal, value = node.gs.compute_valid_moves_and_check_terminal()
            is_white_player_terminated = node.gs.white_to_move
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(node.gs.get_encoded_state(), device=self.model.device).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = node.gs.get_single_valid_moves()
                self.apply_valid_move_mask_to_policy(policy, valid_moves, add_noise=False)
                value = value.item()
                self.expand(node, policy, valid_moves)
            self.backpropagate(node, value, is_white_player_terminated)

        action_probs_all = np.zeros(gs.action_size)
        valid_move_idx = [child.action_taken_id for child in root.children]
        action_probs_all[valid_move_idx] = [child.visit_count for child in root.children]
        action_probs_all /= np.sum(action_probs_all)
        return action_probs_all



    @torch.no_grad()
    def parallel_search(self, game_sessions: list):
        """
        Performs the parallel MCTS search from the given game registers and backpropagates the values.

        :param game_sessions: The list of game registers for parallel processing.
        """
        encoded_states = np.stack([game_session.gs.get_encoded_state() for game_session in game_sessions])
        policy, _ = self.model(torch.tensor(encoded_states, dtype=torch.float32, device=self.model.device))
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        for i, game_session in enumerate(game_sessions):
            game_session.root = Node(self.args, game_session.gs)
            game_session.gs.compute_if_needed_and_get_single_valid_moves()
            valid_moves = game_session.gs.compute_and_get_single_valid_moves()
            game_session_policy = policy[i]
            self.apply_valid_move_mask_to_policy(game_session_policy, valid_moves, True)
            self.expand(game_session.root, game_session_policy, valid_moves)
            self.backpropagate(game_session.root, 0, True)
        for search in range(self.args['num_searches']):
            for game_session in game_sessions:
                game_session.node = None
                node = self.select_leaf_node(game_session.root)
                is_terminal, value = node.gs.compute_valid_moves_and_check_terminal()
                is_white_player_terminated = node.gs.white_to_move
                if is_terminal:
                    self.backpropagate(node, value, is_white_player_terminated)
                else:
                    game_session.node = node
            expandable_grNodes = [mappingIdx for mappingIdx in range(len(game_sessions)) if game_sessions[mappingIdx].node is not None]
            if len(expandable_grNodes) > 0:
                encoded_states = np.stack([game_sessions[mappingIdx].node.gs.get_encoded_state() for mappingIdx in expandable_grNodes])
                policy, value = self.model(torch.tensor(encoded_states, dtype=torch.float32, device=self.model.device))
                policy = torch.softmax(policy, axis=1).cpu().numpy()
            for i, mappingIdx in enumerate(expandable_grNodes):
                node = game_sessions[mappingIdx].node
                valid_moves = node.gs.compute_and_get_single_valid_moves()
                game_session_policy, gr_value = policy[i], value[i]
                self.apply_valid_move_mask_to_policy(game_session_policy, valid_moves, False)
                self.expand(node, game_session_policy, valid_moves)
                self.backpropagate(node, gr_value, node.gs.white_to_move)



