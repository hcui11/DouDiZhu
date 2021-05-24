import numpy as np
from doudizhu import Game, Play, CARD_STR
from mcts import MonteCarloTreeSearchNode
from pg import PGAgent
from train_pg import *

def hand_to_string(cards): #14-vector format to list of card strings

    str = '['
    for index, num in enumerate(cards):
        for i in range(num):
            str += CARD_STR[index] + ", "
    str = str[:-2]
    str += "]"
    return str
def indices_to_string(cards): #list of indices to list of card strings
    if cards == []:
        return '[ ]'
    str = '['
    for index in cards:
        str += CARD_STR[index] + ", "
    str = str[:-2]
    str += "]"
    return str
def main():
    game = Game()
    state = Game(hands=game.hands+0)
    landlordAI = MonteCarloTreeSearchNode(state, 0)
    agent = PGAgent(learning_rate=0.01, device='cpu')
    Naive = NaiveGreedy()
    Random = RandomPlayer()
    Smart = SmartGreedy()


    load_model(agent.model, "PG_param.pth")
    all_players = ["MonteCarlo", "PGAgent", "Naive", "Random", "Smart", "Human"]
    #MonteCarlo can only play as landlord
    players = ["Human", "PGAgent", "Smart"]

    while game.get_winner() == -1:

        player = game.turn
        print(f"PLAYER {game.turn}'s CARDS:")
        print(hand_to_string(game.hands[game.turn]))

        print("Your opponents hand sizes: ", end="")
        for i in range(3):
            if i != game.turn:
                print(sum(game.hands[i]), end=" ")
        print()

        if game.last_move != None:
            print("The play to beat: ", indices_to_string(game.last_move.cards))
        else:
            print("There is no play to beat")

        if players[player] == "Human":
            print("Legal Actions:")
            possible_moves = game.legal_actions()
            for i, action in enumerate(possible_moves):
                print(f'{i}: {indices_to_string(action)}')

        while (True):
            if players[player] == "MonteCarlo":
                landlordAI = landlordAI.best_action()
                landlordAI.parent = None
                print(f"MonteCarlo played {indices_to_string(landlordAI.parent_action.cards)}!")
                input("Press anything to continue")
                game.move(landlordAI.parent_action)
                break
            elif players[player] == "PGAgent" or players[player] == "Smart" or players[player] == "Naive" or players[player] == "Random":
                player = game.turn
                hands = game.hands[player]
                last_move = game.last_move
                last_deal = [] if last_move is None else last_move.cards
                possible_moves = game.legal_actions()
                played_cards = game.played_cards
                is_last_deal_landlord = int(game.last_move == 0)
                is_landlord = int(game.turn == 0)
                if players[player] == "PGAgent":
                    agent.current_state(hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord)
                    action = Play(agent.play())
                    print(f"PGAgent played {indices_to_string(action.cards)}!")
                elif players[player] == "Smart":
                    Smart.current_state(hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord)
                    action = Play(Smart.play())
                    print(f"Smart Greedy played {indices_to_string(action.cards)}!")
                elif players[player] == "Naive":
                    Naive.current_state(hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord)
                    action = Play(Naive.play())
                    print(f"Naive Greedy played {indices_to_string(action.cards)}!")
                elif players[player] == "Random":
                    Random.current_state(hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord)
                    action = Play(Random.play())
                    print(f"Random played {indices_to_string(action.cards)}!")

                input("Press anything to continue")
                game.move(action)
                break
            else:
                move = input(
                    "Please enter your indexed move or enter PASS: ")

                if move == "PASS" or move == "P":
                    move = -1
                elif move.isnumeric() and int(move) < len(possible_moves):
                    move = int(move)
                else:
                    print('Invalid Move!')
                    continue

                move = possible_moves[move]
                play = Play(move)
                print(f"You played a {play.type}!")
                input("Press anything to continue")
                game.move(play)
                try:
                    landlordAI = landlordAI.children[move]
                    landlordAI.parent = None
                except:
                    state = Game(hands=game.hands+0,
                                 last_move=game.last_move,
                                 turn=game.turn,
                                 passes=game.passes)
                    landlordAI = MonteCarloTreeSearchNode(state, 0)
                break
        print("\n\n")
    print(f"Player {game.get_winner()}, {players[game.get_winner()]} wins!")


if __name__ == '__main__':
    main()
