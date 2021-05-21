from doudizhu import GameState
import numpy as np

game = GameState()
game.hands[0] = np.array([4] * 14)
game.hands[0, -1] = 2
possible_actions = game.legal_actions()
print(len(possible_actions))
# print(possible_actions[499:3886])
# print(possible_actions[-10:])