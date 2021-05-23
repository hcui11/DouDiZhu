import numpy as np
from doudizhu import GameState, Play


class MonteCarloTreeSearchNode():
    def __init__(self, state, player, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.player = player
        self.children = []
        self._number_of_visits = 0
        self._results = {1: 0, 0: 0}  # 1 for win, 0 for loss
        self._untried_actions = self.state.legal_actions()

    def q(self):
        wins = self._results[1]
        loses = self._results[0]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        state_params = self.state.simulate(Play(action))
        next_state = GameState(*state_params)
        child_node = MonteCarloTreeSearchNode(next_state,
                                              self.player,
                                              parent=self,
                                              parent_action=Play(action))

        self.children.append(child_node)
        return child_node

    def simulate(self):
        current_state = self.state

        while current_state.get_winner() < 0:
            possible_moves = current_state.legal_actions()
            action = possible_moves[np.random.randint(len(possible_moves))]
            state_params = current_state.simulate(Play(action))
            current_state = GameState(*state_params)
        winner = current_state.get_winner()
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
        choices_weights = [(c.q() / c.n()) + c_param *
                           np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def select(self):
        current_node = self
        while current_node.state.get_winner() < 0:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 1000
        for i in range(simulation_no):
            v = self.select()
            reward = v.simulate()
            v.backpropagate(reward)
        return self.best_child(c_param=0.1)

# from random import shuffle
# from typing import Optional, List
# from itertools import combinations

# # PlayType = Optional[Literal['single', 'double', 'triple', 'bomb', 'triple+1', 'triple+2', 'sisters',
# #                             'airplane', 'airplane+1', 'airplane+2', 'quad+1', 'quad+2', 'straight', 'PASS']]

# CARD_STR = {
#     0: '3',
#     1: '4',
#     2: '5',
#     3: '6',
#     4: '7',
#     5: '8',
#     6: '9',
#     7: '10',
#     8: 'J',
#     9: 'Q',
#     10: 'K',
#     11: 'A',
#     12: '2',
#     13: 'JOKER'
# }

# class Play:
#     def __init__(self, cards: List[int]):
#         '''
#         cards: the cards in play in the current round, as a list of integers
#         type: the type of play in the current round, as a string

#         '''
#         self.cards: List[int] = cards
#         self.type: PlayType = None
#         self.get_info(self.cards)


#     def __str__(self):
#         s = ''
#         for card in self.cards:
#             s += CARD_STR[card] + ' '
#         return s


#     def get_info(self, cards: List[int]):
#         if len(cards) == 0:
#             self.type = 'PASS'
#         elif len(cards) == 1:
#             self.type = 'single'
#         elif len(cards) == 2:
#             if cards[0] == 13:
#                 self.type = 'rocket'
#             else:
#                 self.type = 'double'
#         elif len(cards) == 3:
#             self.type = 'triple'
#         elif len(cards) == 4:
#             if cards[0] == cards[1] == cards[2] == cards[3]:
#                 self.type = 'bomb'
#             elif cards[0] == cards[1] == cards[2]:
#                 self.type = 'triple+1'
#         elif len(cards) == 5:
#             if cards[0] == cards[1] == cards[2] and cards[3] == cards[4]:
#                 self.type = 'triple+2'
#             else:
#                 self.type = 'straight'
#         elif len(cards) == 6:
#             if cards[0] == cards[1] and cards[2] == cards[3] and cards[4] == cards[5]:
#                 self.type = 'sisters'
#             elif cards[0] == cards[1] == cards[2] == cards[3]:
#                 self.type = 'quad+1'
#             elif cards[0] == cards[1] == cards[2]:
#                 self.type = 'airplane'
#             else:
#                 self.type = 'straight'
#         elif len(cards) == 8:
#             if cards[0] == cards[1] == cards[2] == cards[3]:
#                 self.type = 'quad+2'
#             elif cards[0] == cards[1] == cards[2]:
#                 self.type = 'airplane+1'
#             else:
#                 self.type = 'straight'
#         elif len(cards) == 10:
#             if cards[0] == cards[1] == cards[2]:
#                 self.type = 'airplane+2'
#             else:
#                 self.type = 'straight'
#         else:
#             self.type = 'straight'


# class GameState:
#     def __init__(self,
#                  hands: Optional[np.array] = None,
#                  last_move: Optional[Play] = None,
#                  turn: int = 0,
#                  passes: int = 0):
#         '''
#         hands: 3 x 14 np array where hands[i][j] is the number of copies player i has of card j
#         last_move: Play object encoding what cards are in play in the current round
#         turn: the current player encoded as 0, 1, 2 (0 is the Landlord)
#         passes: the number of people who have passed this round
#         '''
#         self.hands = hands
#         self.last_move = last_move
#         self.turn = turn
#         self.passes = passes

#         if hands is None:
#             self.hands = np.zeros((3, 14), dtype=int)
#             self.distribute_cards()


#     def distribute_cards(self):
#         deck = []
#         for i in range(54):
#             deck.append(i // 4)
#         shuffle(deck)

#         for i in range(51):
#             self.hands[i // 17, deck[i]] += 1
#         for i in range(51, 54):
#             self.hands[0, deck[i]] += 1


#     def generate_chains(self, cards, min_length, constraint=0):
#         n = len(cards[0]) if cards else 0
#         chains = []
#         chain = []
#         for card in cards:
#             if card[0] != 12 and card[0] != 13:
#                 if len(chain) == 0 or card[0] == chain[-1] + 1:
#                     if constraint and len(chain) == constraint:
#                         chain = chain[n:]
#                     chain.extend(card)
#                 else:
#                     chain = card[:]

#                 if constraint:
#                     if len(chain) == constraint and chain[0] > self.last_move.cards[0]:
#                         chains.append(chain[:])
#                 else:
#                     if len(chain) >= min_length:
#                         chains.append(chain[:])
#                         for i in range(n, len(chain) - min_length + n, n):
#                             chains.append(chain[i:])
#         return chains


#     def legal_actions(self):
#         singles = []
#         doubles = []
#         triples = []
#         quads = []
#         possible_actions = []

#         for i, n in enumerate(self.hands[self.turn]):
#             if n > 0:
#                 singles.append([i])
#             if n > 1:
#                 doubles.append([i, i])
#                 # Rocket
#                 if i == 13:
#                     possible_actions.append([i, i])
#             if n > 2:
#                 triples.append([i, i, i])
#             # Bomb
#             if n == 4:
#                 possible_actions.append([i, i, i, i])
#                 quads.append([i, i, i, i])

#         # No Last Move
#         if not self.last_move:
#             possible_actions.extend(singles)
#             possible_actions.extend(doubles)
#             possible_actions.extend(triples)
#             for triple in triples:
#                 for single in singles:
#                     if triple[0] != single[0]:
#                         possible_actions.append(triple + single)
#                 for double in doubles:
#                     if triple[0] != double[0]:
#                         possible_actions.append(triple + double)
#             possible_actions.extend(self.generate_chains(singles, 5))
#             possible_actions.extend(self.generate_chains(doubles, 6))

#             airplanes = self.generate_chains(triples, 6)
#             possible_actions.extend(airplanes)
#             kickers1 = set([i for i, v in enumerate(self.hands[self.turn]) if v > 0])
#             kickers2 = set([i for i, v in enumerate(self.hands[self.turn]) if v > 1])
#             for airplane in airplanes:
#                 invalids = set(range(airplane[0], airplane[-1] + 1))
#                 valids1 = kickers1 - invalids
#                 valids2 = kickers2 - invalids
#                 for combo in combinations(valids1, len(airplane) // 3):
#                     possible_actions.append(airplane + list(combo))
#                 for combo in combinations(valids2, len(airplane) // 3):
#                     possible_actions.append(airplane + list(combo) * 2)

#             for q in quads:
#                 q_val = q[0]
#                 q_kickers1 = kickers1 - set([q_val])
#                 q_kickers2 = kickers2 - set([q_val])
#                 for combo in combinations(q_kickers1, 2):
#                     possible_actions.append(q + list(combo))
#                 for combo in combinations(q_kickers2, 2):
#                     possible_actions.append(q + list(combo) * 2)

#         # Single Last Move
#         elif self.last_move.type == 'single':
#             for single in singles:
#                 if single[0] > self.last_move.cards[0]:
#                     possible_actions.append(single)
#         # Double Last Move
#         elif self.last_move.type == 'double':
#             for double in doubles:
#                 if double[0] > self.last_move.cards[0]:
#                     possible_actions.append(double)
#         # Triple Last Move
#         elif self.last_move.type == 'triple':
#             for triple in triples:
#                 if triple[0] > self.last_move.cards[0]:
#                     possible_actions.append(triple)
#         # Triple+1 Last Move
#         elif self.last_move.type == 'triple+1':
#             for triple in triples:
#                 if triple[0] > self.last_move.cards[0]:
#                     for single in singles:
#                         if triple[0] != single[0]:
#                             possible_actions.append(triple + single)
#         # Triple+2 Last Move
#         elif self.last_move.type == 'triple+2':
#             for triple in triples:
#                 if triple[0] > self.last_move.cards[0]:
#                     for double in doubles:
#                         if triple[0] != double[0]:
#                             possible_actions.append(triple + double)
#         # Straight Last Move
#         elif self.last_move.type == 'straight':
#             n = len(self.last_move.cards)
#             possible_actions.extend(self.generate_chains(singles, 5, constraint=n))
#         # Sisters Last Move
#         elif self.last_move.type == 'sisters':
#             n = len(self.last_move.cards)
#             possible_actions.extend(self.generate_chains(doubles, 6, constraint=n))
#         # Airplane Last Move
#         elif 'airplane' in self.last_move.type:
#             n = len(self.last_move.cards)
#             if self.last_move.type[-1].isnumeric():
#                 k = int(self.last_move.type[-1])
#                 n -= n * k // (3 + k)
#             airplanes = self.generate_chains(triples, 6, constraint=n)

#             # Airplane+1 Last Move
#             if self.last_move.type == 'airplane+1':
#                 kickers = set([i for i, v in enumerate(self.hands[self.turn]) if v > 0])
#                 for airplane in airplanes:
#                     invalids = set(range(airplane[0], airplane[-1] + 1))
#                     valids = kickers - invalids
#                     for combo in combinations(valids, n // 3):
#                         possible_actions.append(airplane + list(combo))
#             # Airplane+2 Last Move
#             elif self.last_move.type == 'airplane+2':
#                 kickers = set([i for i, v in enumerate(self.hands[self.turn]) if v > 1])
#                 for airplane in airplanes:
#                     invalids = set(range(airplane[0], airplane[-1] + 1))
#                     valids = kickers - invalids
#                     for combo in combinations(valids, n // 3):
#                         possible_actions.append(airplane + list(combo) * 2)
#             else:
#                 possible_actions.extend(airplanes)
#         elif 'quad' in self.last_move.type:
#             n = len(self.last_move.cards)
#             num_kicker_pairs = int(self.last_move.type[-1])

#             curr_val = self.last_move.cards[0]

#             possible_kickers = set([card for card, num in enumerate(self.hands[self.turn]) if num >= num_kicker_pairs])
#             for q in quads:
#                 q_val = q[0]
#                 if q_val > curr_val:
#                     q_kickers = possible_kickers - set([q_val])
#                     for combo in combinations(q_kickers, 2):
#                         possible_actions.append(q + list(combo) * num_kicker_pairs)

#         possible_actions.append([])
#         return possible_actions


#     def get_winner(self):
#         for i in range(3):
#             if sum(self.hands[i]) == 0:
#                 return i
#         return -1


#     def move(self, play):
#         if play.cards:
#             self.last_move = play
#             for card in play.cards:
#                 self.hands[self.turn, card] -= 1
#             self.passes = 0
#         else:
#             self.passes += 1
#             if self.passes == 2:
#                 self.passes = 0
#                 self.last_move = None
#         self.turn = (self.turn + 1) % 3


#     def simulate(self, play):
#         hands = self.hands + 0
#         if play.cards:
#             for card in play.cards:
#                 hands[self.turn, card] -= 1
#             passes = 0
#         else:
#             play = self.last_move
#             passes = self.passes + 1
#             if self.passes == 2:
#                 passes = 0
#                 play = None
#         turn = (self.turn + 1) % 3
#         return hands, play, turn, passes
