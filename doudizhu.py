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
            elif cards[1] == cards[2] == cards[3]:
                self.type = "triple+1"
                self.main_rank = cards[1]
        elif len(cards) == 5:
            if cards[0] == cards[1] == cards[2] and cards[3] == cards[4]:
                self.type = "triple+2"
                self.main_rank = cards[0]
            elif cards[2] == cards[3] == cards[4] and cards[0] == cards[1]:
                self.type = "triple+2"
                self.main_rank = cards[2]
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
                 turn=0):
        self.hands = hands
        self.last_move = last_move
        self.turn = turn
        if not hands:
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
                if len(straight) == 0 or single[0] == straight[-1] + 1:
                    straight.append(single[0])
                else:
                    straight = [single[0]]
                if len(straight) > 4 and straight[0] > self.last_move.cards[0]:
                    possible_actions.append(straight[:])

        possible_actions.append([-1])
        return possible_actions


    def over(self):
        for i in range(3):
            if sum(self.hands[i]) == 0:
                return i
        return -1

    def move(self, play):
        self.last_move = play
        for card in play.cards:
            self.hands[self.turn, card] -= 1


def main():
    game = Game()
    pass_counter = 0
    while game.over() == -1:
        if pass_counter == 2:
            game.last_move = None
            pass_counter = 0

        print(f"PLAYER {game.turn}'s CARDS:")
        print(game.hands[game.turn])

        print("Your opponents hand sizes: ", end="")
        for i in range(3):
            if i != game.turn:
                print(sum(game.hands[i]), end=" ")
        print()

        if game.last_move != None:
            print("The play to beat: ", game.last_move.cards)
        else:
            print("There is no play to beat")

        print("Legal Actions:")
        possible_moves = game.legal_actions()
        for i, action in enumerate(possible_moves[:-1]):
            print(f'{i}: {action}')
        
        while (True):
            move = input(
                "Please enter your indexed move or enter PASS: ")
            if move == "PASS" or move == "P":
                pass_counter += 1
                break
            
            if move.isnumeric() and int(move) < len(possible_moves):
                move = possible_moves[int(move)]
                play = Play(move)
                pass_counter = 0
                print(f"You played a {play.type}!")
                input("Press anything to continue")
                game.move(play)
                break
        game.turn = (game.turn + 1) % 3
        print("\n\n")
    print(f"Player {game.over()} wins!")

if __name__ == '__main__':
    main()