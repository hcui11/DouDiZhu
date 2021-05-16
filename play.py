import numpy as np
from doudizhu import Game, Play
from mcts import MonteCarloTreeSearchNode

def main():
    game = Game()
    state = Game(hands=game.hands+0)
    landlordAI = MonteCarloTreeSearchNode(state, 0)

    while game.over() == -1:
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
            if game.turn == 0:
                landlordAI = landlordAI.best_action()
                landlordAI.parent = None
                print(f"Landlord played a {landlordAI.parent_action.type}!")
                input("Press anything to continue")
                game.move(landlordAI.parent_action)
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
                
                landlordAI = landlordAI.children[move]
                landlordAI.parent = None
                move = possible_moves[move]
                play = Play(move)
                print(f"You played a {play.type}!")
                input("Press anything to continue")
                game.move(play)
                break
        print("\n\n")
    print(f"Player {game.over()} wins!")

if __name__ == '__main__':
    main()