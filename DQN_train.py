from DQN_agent import Agent
import numpy as np
from doudizhu import GameState, Play, CARD_STR
from mcts import MonteCarloTreeSearchNode
import utils
import pickle
import matplotlib.pyplot as plt

with open('action_encoder.pt', 'rb') as f:
  encoded_actions = pickle.load(f)
decoded_actions = {i: a for a, i in encoded_actions.items()}

def main():
  n_games = 1000
  gamma = 0.01
  epsilon = 0.8
  lr = 0.001
  input_dims = 32
  batch_size = 64
  n_actions = len(encoded_actions)
  
  LandlordAI = Agent(gamma, epsilon, lr, [input_dims], batch_size, n_actions)
  PeasantAI = Agent(gamma, epsilon, lr, [input_dims], batch_size, n_actions)
  
  LandlordAI_wins = 0
  PeasantAI_wins = 0
  
  LandlordAI_winRates = []
  PeasantAI_winRates = []
  
  for i in range(n_games):
    if i % 50 == 0:
      print("game ", str(i))
    game = GameState()
    while game.get_winner() == -1:
      turn = game.turn
      observation = game.get_player_state(turn)
      possible_moves = game.legal_actions()
      possible_moves_indices = np.array([encoded_actions[tuple(a)] for a in possible_moves])
  
      if turn == 0:
        action = LandlordAI.choose_action(observation, possible_moves_indices)
        game.move(Play(decoded_actions[action]))
        observation_ = game.get_player_state(turn)
        if game.get_winner() != -1:
          if game.get_winner() == 0:
            reward = 1
            LandlordAI_wins += 1
          else:
            reward = -1
          done = True
        else:
          reward = 0
          done = False
        LandlordAI.store_transition(observation, action, reward, observation_, done)
        LandlordAI.learn()        
      
      else:
        action = PeasantAI.choose_action(observation, possible_moves_indices)
        game.move(Play(decoded_actions[action]))
        observation_ = game.get_player_state(turn)
        if game.get_winner() != -1:
          if game.get_winner() == 0:
            reward = -1
          else:
            reward = 1
            PeasantAI_wins += 1
          done = True
        else:
          reward = 0
          done = False
        PeasantAI.store_transition(observation, action, reward, observation_, done)
        PeasantAI.learn() 
        
    LandlordAI_winRates.append(LandlordAI_wins / (i + 1))
    PeasantAI_winRates.append(PeasantAI_wins / (i + 1))
  
  plt.plot(LandlordAI_winRates)
  plt.plot(PeasantAI_winRates)
  plt.legend(['Landlord (DQN)', 'Peasant (DQN)'])
  plt.title('Win Rate vs. Games Played')
  plt.savefig('Win Rate vs. Games Played (DQN Landlord, DQN Peasant).png')

  print("Landlord Final Win Rate: ", str(LandlordAI_winRates[-1]))
  print("Peasant Final Win Rate: ", str(PeasantAI_winRates[-1]))      
    
  
  
if __name__ == '__main__':
    main()