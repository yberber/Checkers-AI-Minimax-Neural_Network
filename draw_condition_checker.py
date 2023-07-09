
import move_id_mapper
from checkers_engine import CheckersGameState

class Draw_Condition_Checker:

    def __init__(self, undo_possible=False):
        self.saved_move_and_encoded_state = []

        self.undo_possible = undo_possible
        if undo_possible:
            self.saved_move_ids = []

    def add_move_state(self, gs:CheckersGameState):
        move = gs.move_log[-1]
        if not self.undo_possible:
            if move.captured_piece == "--":
                self.saved_move_and_encoded_state.append((move.to_checkers_notation(), gs.get_encoded_state()[:4]))
            else:
                self.reset()
        else:
            if move.captured_piece == "--":
                self.saved_move_and_encoded_state.append((move.to_checkers_notation(), gs.get_encoded_state()[:4]))
                self.saved_move_ids.append(gs.move_count)

    def update_move_entries_after_undo(self, gs:CheckersGameState):
        current_move_id = gs.move_count

        for entry_id in reversed(range(len(self.saved_move_ids))):
            if self.saved_move_ids[entry_id] > current_move_id:
                self.saved_move_ids.pop(entry_id)
                self.saved_move_and_encoded_state.pop(entry_id)
            else:
                break


    def reset(self):
        self.saved_move_and_encoded_state = []

    def check_threefold_repetition(self):
        if len(self.saved_move_and_encoded_state) >= 5:
            last_move, last_encoded_state = self.saved_move_and_encoded_state[-1]
            occurrences = 1
            for move, encoded_state in self.saved_move_and_encoded_state[:-1]:
                if move == last_move and (encoded_state == last_encoded_state).all():
                    occurrences += 1
                if occurrences >= 3:
                    return True
        return False

    def remove_last_move_state(self):
        self.saved_move_and_encoded_state.pop()



