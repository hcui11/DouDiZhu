import numpy as np
from doudizhu import GameState, Play, CARD_STR
from mcts import MonteCarloTreeSearchNode


def main():
    game = GameState()
    state = GameState(hands=game.hands+0)
    landlordAI = MonteCarloTreeSearchNode(state, 0)

    while game.get_winner() == -1:
        print(f'PLAYER {game.turn}\'s CARDS:')
        hand_str = ''
        for i, n in enumerate(game.hands[game.turn]):
            hand_str += ' '.join([CARD_STR[i]] * int(n)) + ' '
        print(hand_str)

        print('Your opponents hand sizes: ', end='')
        for i in range(3):
            if i != game.turn:
                print(sum(game.hands[i]), end=' ')
        print()

        if game.last_move != None:
            print('The play to beat: ', game.last_move)
        else:
            print('There is no play to beat')

        print('Legal Actions:')
        possible_moves = game.legal_actions()
        for i, action in enumerate(possible_moves[:-1]):
            print(f'{i}: {[CARD_STR[c] for c in action]}')

        while (True):
            if game.turn == 0:
                landlordAI = landlordAI.best_action()
                landlordAI.parent = None
                print(f'Landlord played a {landlordAI.parent_action.type}!')
                print(landlordAI.parent_action)
                input('Press anything to continue')
                game.move(landlordAI.parent_action)
                break
            else:
                move = input(
                    'Please enter your indexed move or enter PASS: ')

                if move == 'PASS' or move == 'P':
                    move = -1
                elif move.isnumeric() and int(move) < len(possible_moves):
                    move = int(move)
                else:
                    print('Invalid Move!')
                    continue

                move = possible_moves[move]
                play = Play(move)
                print(f'You played a {play.type}!')
                input('Press anything to continue')
                game.move(play)
                try:
                    landlordAI = landlordAI.children[move]
                    landlordAI.parent = None
                except:
                    state = GameState(hands=game.hands+0,
                                 last_move=game.last_move,
                                 turn=game.turn,
                                 passes=game.passes)
                    landlordAI = MonteCarloTreeSearchNode(state, 0)
                break
        print('\n\n')
    print(f'Player {game.get_winner()} wins!')


if __name__ == '__main__':
    main()
