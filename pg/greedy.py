# -*- coding: utf-8 -*-
"""
Created on Sat May 22 15:12:42 2021

@author: 46012
"""

import numpy as np
from doudizhu import Game, Play

class NaiveGreedy():
    def __init__(self):
        pass

    def current_state(self, hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord):
        self.possible_actions = possible_moves

    def play(self):
        return self.possible_actions[0]

class SmartGreedy():
    def __init__(self, temp = True):
        self.temp = temp
        pass

    def current_state(self, hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord):
        self.possible_actions = possible_moves
        self.hands = hands
        self.last_deal = last_deal
        self.play_info = Play(last_deal)
        self.played_cards = played_cards

    def current_state2(self, info, possible_moves):
        self.hands = info[:14]
        last_deal = info[14:28]
        self.last_deal = []
        for i in range(len(last_deal)):
            for j in range(last_deal[i]):
                self.last_deal.append(i)
        self.play_info = Play(last_deal)
        self.possible_actions = possible_moves


    def play(self):
        # print("\n")
        # print(self.hands)
        # print("last play", self.last_deal)

        best = np.count_nonzero(self.hands)
        sum = np.sum([i*j for i,j in enumerate(self.hands)])
        hand_size= np.sum(self.hands)
        best_arg = 0
        tie_break = 0

        if self.play_info.type == 'PASS':
            for i, num in enumerate(self.hands):
                if num>0 and num<4:
                    #print([i for j in range(num)])
                    return [i for j in range(num)]

        for i, action in enumerate(self.possible_actions):
            hand_sum = 0
            for card in action:
                hand_sum += card
                self.hands[card] -= 1
            num_cards = np.count_nonzero(self.hands)
            if num_cards == 0:
                for card in action:
                    self.hands[card] += 1
                return action
            if num_cards < best:
                best = num_cards
                best_arg = i
                tie_break = (sum - hand_sum)/(hand_size - len(action))
                if Play(action).type != "bomb": #punishment for using a bomb
                    tie_break -= 3
            elif self.temp and num_cards == best:
                new_tiebreak = (sum - hand_sum)/(hand_size - len(action))
                if Play(action).type != "bomb": #punishment for using a bomb
                    new_tiebreak -= 3
                if new_tiebreak > tie_break and Play(action).type != "PASS":
                    #print("comparing", self.possible_actions[best_arg], self.possible_actions[i])
                    best_arg = i
                    tie_break = new_tiebreak
            for card in action:
                self.hands[card] += 1
        # print(self.possible_actions[best_arg])
        #
        # print("tiebreak is ", tie_break)
        #print("play made", self.possible_actions[best_arg])
        return(self.possible_actions[best_arg])


class RandomPlayer():
    def __init__(self):
        pass

    def current_state(self, hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord):
        self.possible_actions = possible_moves

    def play(self):
        return self.possible_actions[np.random.choice(len(self.possible_actions))]
