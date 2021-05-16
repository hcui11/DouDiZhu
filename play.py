import numpy as np
from doudizhu import Game, Play
from mcts import MonteCarloTreeSearchNode

def main():
    game = Game()
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
            move = input(
                "Please enter your indexed move or enter PASS: ")
            if move == "PASS" or move == "P":
                move = possible_moves[-1]
                play = Play(move)
                game.move(play)
                break
            
            if move.isnumeric() and int(move) < len(possible_moves):
                move = possible_moves[int(move)]
                play = Play(move)
                print(f"You played a {play.type}!")
                input("Press anything to continue")
                game.move(play)
                break
        print("\n\n")
    print(f"Player {game.over()} wins!")

if __name__ == '__main__':
    main()