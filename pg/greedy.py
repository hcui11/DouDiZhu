# -*- coding: utf-8 -*-
"""
Created on Sat May 22 15:12:42 2021

@author: 46012
"""

import numpy as np

class NaiveGreedy():
    def __init__(self):
        pass

    def current_state(self, hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord):
        self.possible_actions = possible_moves

    def play(self):
        return self.possible_actions[0]

class RandomPlayer():
    def __init__(self):
        pass

    def current_state(self, hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord):
        self.possible_actions = possible_moves

    def play(self):
        return self.possible_actions[np.random.choice(len(self.possible_actions))]