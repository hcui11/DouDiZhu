import numpy as np
from doudizhu import Game, Play
from mcts import MonteCarloTreeSearchNode
from pg import PGAgent
from train_pg import *


def main():
    game = Game()
    state = Game(hands=game.hands+0)
    landlordAI = MonteCarloTreeSearchNode(state, 0)
    agent = PGAgent(learning_rate=0.01, device='cpu')
    load_model(agent.model, "PG_param.pth")

    while game.get_winner() == -1:
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
            # if game.turn == 0:
            #     landlordAI = landlordAI.best_action()
            #     landlordAI.parent = None
            #     print(f"Landlord played a {landlordAI.parent_action.type}!")
            #     input("Press anything to continue")
            #     print(type(landlordAI.parent_action))
            #     game.move(landlordAI.parent_action)
            #     break
            if game.turn == 0:
                player = game.turn
                hands = game.hands[player]
                last_move = game.last_move
                last_deal = [] if last_move is None else last_move.cards
                possible_moves = game.legal_actions()
                played_cards = game.played_cards
                is_last_deal_landlord = int(game.last_move == 0)
                is_landlord = int(game.turn == 0)
                agent.current_state(hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord)
                action = Play(agent.play())
                print(f"Landlord played a {action.type}!")
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
    print(f"Player {game.get_winner()} wins!")


if __name__ == '__main__':
    main()
