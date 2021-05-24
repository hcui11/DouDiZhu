import numpy as np
from doudizhu import GameState, Play, CARD_STR
#from ..alpha_zero.Game import Game
from mcts import MonteCarloTreeSearchNode
from pg import PGAgent
from supervised import Supervised
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from tqdm import trange
import random
from greedy import NaiveGreedy, RandomPlayer, SmartGreedy
from visdom import Visdom
from copy import deepcopy
from train_pg import *
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', 'alpha_zero')))
from Game import Game

inputs = []
target = []
G = Game()
inv_map = {v: k for k, v in G.decoded_actions.items()}

def start_game(players, info=False, save_data = False):
    game = GameState()
    while game.get_winner() == -1:
        player = game.turn
        hands = game.hands[player]
        last_move = game.last_move
        last_deal = [] if last_move is None else last_move.cards
        possible_moves = game.legal_actions()
        played_cards = game.played_cards
        is_last_deal_landlord = int(game.last_move == 0)
        is_landlord = int(game.turn == 0)
        last_move = game.last_move
        last_deal = [] if last_move is None else last_move.cards

        possible_move_indices = [inv_map[tuple(i)] for i in possible_moves]

        if save_data:
            inputs.append(game.get_player_state(player))

        if type(players[player]) == Supervised:
            action = G.decoded_actions[int(players[player].play(torch.FloatTensor(game.get_player_state(player)), possible_move_indices))]
            action = list(action)
        else:
            players[player].current_state(hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord)
            action = players[player].play()

        #print(action)
        if game.turn == 0 and info:
            print("supervised:", action)
            players[1].current_state(hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord)
            action2 = players[1].play()
            print("correct:", action2)
        #print(game.turn)

        if save_data:
            target.append(inv_map[tuple(action)])
        # print(action)
        # print(G.decoded_actions[inv_map[tuple(action)]])
        #action = players[player].play(game.legal_actions(), player, game.hands[player], last_deal)
        play = Play(action)
        if info:
            print(hands)
            print(last_deal)
            print(f'player {game.turn}:', action)
            print()
        game.move(play)
    if info:
        print(f"Player {game.get_winner()} wins!")
    return game.get_winner()

def save_model(model, path):
    model_saving_path = path
    print('saving model to ' + model_saving_path)
    torch.save(model.state_dict(), model_saving_path)

def load_model(model, path):
    model_loading_path = path
    print('loading model from ' + model_loading_path)
    model.load_state_dict(torch.load(model_loading_path))
    model.train()

if __name__ == '__main__':

    Smart = SmartGreedy()
    Random = RandomPlayer()
    Naive = NaiveGreedy()
    players = [Smart, Smart, Smart]

    for i in range(1000):
        start_game(players, save_data = True)

    agent = Supervised(G)
    load_model(agent, "Super_param.pth")




    players = [agent, Random, Random]
    # start_game(players, info = True)
    # sys.exit()
    wins = [0,0,0]

    for i in range(100):
        wins[start_game(players)] += 1
    print(wins)
    #sys.exit()


    inputs = torch.FloatTensor(inputs)
    target = torch.FloatTensor(target)
    target = F.one_hot(target.to(torch.int64), num_classes = 8542)
    # for t in target:
    #     if t[8541] == 0:
    #          t[:8541] = -100

    opt = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(30):
        outputs = agent(inputs)
        loss = criterion(outputs, target.float())
        #print(loss.dtype)
        loss.backward()
        opt.step()
        print(loss)
    save_model(agent, "Super_param.pth")

    players = [agent, Random, Random]
    wins = [0,0,0]
    for i in range(100):
        wins[start_game(players)] += 1
    print(wins)
    sys.exit()

    sys.exit()
