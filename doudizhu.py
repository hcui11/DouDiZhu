import random
import numpy as np
class Play:
    # types:
    #single, double, triple, bomb, triple+1
    def __init__(self, cards):
        self.cards = cards
        self.type = None
        self.main_rank = None
        self.get_info(self.cards)

    def __str__(self):
        s = ""
        for card in self.cards:
            s += str(card) + "\n"
        return s

    def get_info(self, cards):
        # print(cards)
        if len(cards) == 1:
            self.type = "single"
            self.main_rank = cards[0]
        elif len(cards) == 2:
            if cards[0] == cards[1]:
                self.type = "double"
                self.main_rank = cards[0]
            else:
                self.type = "invalid"
        elif len(cards) == 3:
            if cards[0] == cards[1] == cards[2]:
                self.type = "triple"
                self.main_rank = cards[0]
            else:
                self.type = "invalid"
        elif len(cards) == 4:
            if cards[0] == cards[1] == cards[2] == cards[3]:
                self.type = "bomb"
                self.main_rank = cards[0]
            elif cards[0] == cards[1] == cards[2]:
                self.type = "triple+1"
                self.main_rank = cards[0]
        elif len(cards) == 5:
            if cards[0] == cards[1] == cards[2] and cards[3] == cards[4]:
                self.type = "triple+2"
                self.main_rank = cards[0]
            else:
                for i in range(1, len(cards)):
                    if cards[i] != cards[i-1] + 1:
                        self.type = "invalid"
                        return
                    self.type = "straight"
                    self.main_rank = cards[0]

    def beats_hand(self, other):
        if other.type == "bomb":
            if self.type == "bomb" and self.main_rank > other.main_rank:
                return True
            return False
        if self.type == "bomb":
            return True

        # both normal hands
        if self.type != other.type:
            return False
        return self.main_rank > other.main_rank

class Game:
    def __init__(self,
                 hands=None,
                 last_move=None,
                 turn=0,
                 passes=0):
        self.hands = hands
        self.last_move = last_move
        self.turn = turn
        self.passes = passes
        if hands is None:
            self.hands = np.zeros((3, 14))
            self.distribute_cards()

    def distribute_cards(self):
        deck = []
        for i in range(54):
            deck.append(i // 4)
        random.shuffle(deck)

        for i in range(51):
            self.hands[i // 17, deck[i]] += 1
        for i in range(51, 54):
            self.hands[0, deck[i]] += 1

    def legal_actions(self):
        singles = []
        doubles = []
        triples = []
        possible_actions = []

        for i, n in enumerate(self.hands[self.turn]):
            if n > 0:
                singles.append([i])
            if n > 1:
                doubles.append([i, i])
                if i == 13:
                    possible_actions.append([i, i])
            if n > 2:
                triples.append([i, i, i])
            if n == 4:
                possible_actions.append([i, i, i, i])
        
        # No Last Move
        if not self.last_move:
            possible_actions.extend(singles)
            possible_actions.extend(doubles)
            possible_actions.extend(triples)
            for triple in triples:
                for single in singles:
                    if triple[0] != single[0]:
                        possible_actions.append(triple + single)
                for double in doubles:
                    if triple[0] != double[0]:
                        possible_actions.append(triple + double)
            straight = []
            for single in singles:
                if single[0] != 12 and single[0] != 13:
                    if len(straight) == 0 or single[0] == straight[-1] + 1:
                        straight.append(single[0])
                    else:
                        straight = [single[0]]
                    if len(straight) > 4:
                        possible_actions.append(straight[:])
        # No Last Move or Single Last Move
        elif self.last_move.type == "single":
            for single in singles:
                if single[0] > self.last_move.cards[0]:
                    possible_actions.append(single)
        # No Last Move or Double Last Move
        elif self.last_move.type == "double":
            for double in doubles:
                if double[0] > self.last_move.cards[0]:
                    possible_actions.append(double)
        # No Last Move or Triple Last Move
        elif self.last_move.type == "triple":
            for triple in triples:
                if triple[0] > self.last_move.cards[0]:
                    possible_actions.append(triple)
        # No Last Move or Triple+1 Last Move
        elif self.last_move.type == "triple+1":
            for triple in triples:
                if triple[0] > self.last_move.cards[0]:
                    for single in singles:
                        if triple[0] != single[0]:
                            possible_actions.append(triple + single)
        # No Last Move or Triple+2 Last Move
        elif self.last_move.type == "triple+2":
            for triple in triples:
                if triple[0] > self.last_move.cards[0]:
                    for double in doubles:
                        if triple[0] != double[0]:
                            possible_actions.append(triple + double)
        # No Last Move or Straight Last Move
        elif self.last_move.type == "straight":
            straight = []
            for single in singles:
                if single[0] != 12 and single[0] != 13:
                    if len(straight) == 0 or single[0] == straight[-1] + 1:
                        straight.append(single[0])
                    else:
                        straight = [single[0]]
                    if len(straight) > 4 and straight[0] > self.last_move.cards[0]:
                        possible_actions.append(straight[:])

        possible_actions.append([])
        return possible_actions


    def over(self):
        for i in range(3):
            if sum(self.hands[i]) == 0:
                return i
        return -1

    def move(self, play):
        if play.cards:
            self.last_move = play
            for card in play.cards:
                self.hands[self.turn, card] -= 1
            self.passes = 0
        else:
            self.passes += 1
            if self.passes == 2:
                self.passes = 0
                self.last_move = None
        self.turn = (self.turn + 1) % 3

    def simulate(self, play):
        hands = self.hands + 0
        if play.cards:
            for card in play.cards:
                hands[self.turn, card] -= 1
            passes = 0
        else:
            play = self.last_move
            passes = self.passes + 1
            if self.passes == 2:
                passes = 0
                play = None
        turn = (self.turn + 1) % 3
        return hands, play, turn, passes
