import numpy as np
from doudizhu import Game, Play

class MonteCarloTreeSearchNode():
    def __init__(self, state, player, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.player = player
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = {1: 0, 0: 0} # 1 for win, 0 for loss
        self._untried_actions = self.state.legal_actions()

    def q(self):
        wins = self._results[1]
        loses = self._results[0]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.move(Play(action))
        child_node = MonteCarloTreeSearchNode(next_state,
                                              parent=self,
                                              parent_action=action)

        self.children.append(child_node)
        return child_node

    def simulate(self):
        current_state = self.state
        
        while current_state.over() < 0:
            possible_moves = current_state.legal_actions()
            action = possible_moves[np.random.randint(len(possible_moves))]
            current_state = current_state.move(Play(action))
        winner = current_state.over()
        if self.player == 0:
            return int(winner == 0)
        else:
            return int(winner == 1 or winner == 2)

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    def best_child(self, c_param=0.1): 
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def select(self):
        current_node = self
        while current_node.state.over() < 0:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
    def best_action(self):
        simulation_no = 100
        for i in range(simulation_no):
            v = self.select()
            reward = v.simulate()
            v.backpropagate(reward)
        return self.best_child(c_param=0.1)

def main():
    root = MonteCarloTreeSearchNode(state = initial_state)
    selected_node = root.best_action()
    return