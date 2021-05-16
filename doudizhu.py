import random
import numpy as np
class Play:
    # types:
    #single, double, triple, bomb, triple+1
    def __init__(self, cards):
        self.cards = sorted(cards)
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
            if n > 2:
                doubles.append([i, i, i])
            if n == 4:
                possible_actions.append([i, i, i, i])
        # No Last Move or Single Last Move
        if not self.last_move or self.last_move.type == "single":
            possible_actions.extend(singles)
        # No Last Move or Double Last Move
        if not self.last_move or self.last_move.type == "double":
            possible_actions.extend(doubles)
        # No Last Move or Triple Last Move
        if not self.last_move or self.last_move.type == "double":
            possible_actions.extend(triples)
        # No Last Move or Triple+1 Last Move
        if not self.last_move or self.last_move.type == "triple+1":
            for triple in triples:
                for single in singles:
                    possible_actions.append(triple + single)
        # No Last Move or Triple+2 Last Move
        if not self.last_move or self.last_move.type == "triple+2":
            for triple in triples:
                for double in doubles:
                    possible_actions.append(triple + double)
        # No Last Move or Straight Last Move
        if not self.lat_move or self.last_move.type == "straight":
            straight = []
            for single in singles:
                if len(straight) == 0 or single == straight[-1]:
                    straight.append(single)
                else:
                    straight = []
                if len(straight) > 4:
                    possible_actions.append(straight[:])

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
        
        while (True):
            move = input(
                "Please enter your move as a list of numbers separated by a space or enter PASS: ")
            if move == "PASS" or move == "P":
                pass_counter += 1
                break

            move = [int(i) for i in move.split()]
            play = Play(move)
            if play.type == "invalid":
                print("Your move is invalid")
            elif game.last_move != None and not play.beats_hand(game.last_move):
                print("Your move does not beat the last move played")
            else:
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