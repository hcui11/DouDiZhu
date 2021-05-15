import numpy as np
class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit_num = suit
        suits = ["hearts", "diamonds", "clubs", "spades"]

        self.suit = suits[suit]

    def __str__(self):
        return str(self.rank) + " of " + self.suit
    def __repr__(self):
        return str(self)

class Hand:
    #types:
    #single, double, triple, bomb, triple+1
    def __init__(self, cards):
        self.cards = sorted(cards, key = lambda x: x.rank)
        self.type = None
        self.main_rank = None
        self.get_info(self.cards)
    def __str__(self):
        s = ""
        for card in self.cards:
            s += str(card) + "\n"
        return s

    def get_info(self, cards):
        #print(cards)
        if len(cards) == 1:
            self.type = "single"
            self.main_rank = cards[0].rank
        elif len(cards) == 2:
            if cards[0].rank == cards[1].rank:
                self.type = "double"
                self.main_rank = cards[0].rank
            else:
                self.type = "invalid"
        elif len(cards) == 3:
            if cards[0].rank == cards[1].rank == cards[2].rank:
                self.type = "triple"
                self.main_rank = cards[0].rank
            else:
                self.type = "invalid"
        elif len(cards) == 4:
            if cards[0].rank == cards[1].rank == cards[2].rank == cards[3].rank:
                self.type = "bomb"
                self.main_rank = cards[0].rank
            elif cards[0].rank == cards[1].rank == cards[2].rank:
                self.type = "triple+1"
                self.main_rank = cards[0].rank
            elif cards[1].rank == cards[2].rank == cards[3].rank:
                self.type = "triple+1"
                self.main_rank = cards[1].rank
        # elif len(cards) == 5:
        #     if cards[0].rank == cards[1].rank == cards[2].rank or \
        #             cards[1].rank == cards[2].rank == cards[3].rank:
        #         return "triple+1"

    def beats_hand(self, other):
        if other.type == "bomb":
            if self.type == "bomb" and self.main_rank > other.main_rank:
                    return True
            return False
        if self.type == "bomb":
            return True

        #both normal hands
        if self.type != other.type:
            return False
        return self.main_rank > other.main_rank

class Player:
    def __init__(self, hand):
        self.hand = hand

    def get_move(self):
        pass

class Game:
    def __init__(self):

        self.hands = [[] for i in range(3)]
        self.last_move = None
        self.distribute_cards()


    def distribute_cards(self):
        deck = []
        for i in range(54):
            deck.append(Card(int(i/4), i%4))
        deck = np.array(deck)
        np.random.shuffle(deck)

        for i in range(51):
            self.hands[int(i/17)].append(deck[i])
        for i in range(51, 54):
            self.hands[0].append(deck[i])
        for i in range(3):
            self.hands[i] = sorted(self.hands[i], key = lambda x: x.rank)

    def over(self):
        for i in range(3):
            if len(self.hands[i]) == 0:
                return i
        return -1

    def play(self):
        turn = 0
        pass_counter = 0
        while self.over() == -1:
            if pass_counter == 2:
                self.last_move = None
                pass_counter = 0
            print(f"PLAYER {turn}'s CARDS:")
            for index, card in enumerate(self.hands[turn]):
                print(str(index) + ":", card)
            print("Your opponents hand sizes: ", end = "")

            for i in range(3):
                if i != turn:
                    print(len(self.hands[i]), end = " ")
            print()
            if self.last_move != None:
                print("The play to beat: ", self.last_move.cards)
            else:
                print("There is no play to beat")
            while (True):
                move = input("Please enter your move as a list of numbers separated by a space or enter PASS: ")
                if move == "PASS" or move == "P":
                    pass_counter += 1
                    break

                move = [int(i) for i in move.split()]
                move.sort()
                play = Hand([self.hands[turn][i] for i in move])
                if play.type == "invalid":
                    print("Your move is invalid")
                elif self.last_move != None and not play.beats_hand(self.last_move):
                    print("Your move does not beat the last move played")
                else:
                    pass_counter = 0
                    print(f"You played a {play.type}!")
                    input("Press anything to continue")
                    self.last_move = play
                    for i in move[::-1]:
                        self.hands[turn].pop(i)
                    break
            turn = (turn + 1) % 3
            print("\n\n")
        print(f"Player {self.over()} wins!")



G = Game()
G.play()
