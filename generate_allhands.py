from doudizhu import GameState
import numpy as np
import pickle

game = GameState()
game.hands[0] = np.array([4] * 14)
game.hands[0, -1] = 2
possible_actions = game.legal_actions()
print(len(possible_actions))
# print(possible_actions[499:3886])
# print(possible_actions[-10:])

possible_actions_t = [tuple(t) for t in possible_actions]
action_encoder = {action: i for i, action in enumerate(possible_actions_t)}
with open('action_encoder.pt', 'wb') as f:
    pickle.dump(action_encoder, f)
