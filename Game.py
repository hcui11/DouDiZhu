import numpy as np
from random import shuffle
from doudizhu import GameState, Play
import pickle

class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self):
        """
        player: 0 for landlord, 1 or 2 for farmers
        state representation: 1D np.ndarray where
            [0:14]: player 0's cards
            [14:28]: player 1's cards
            [28:42]: player 2's cards
            [42:56]: the last move's cards
            [56]: number of passes this round
            [57]: current player at [0:14] in canonical board, current player in normal board
        """
        with open('action_encoder.pt', 'rb') as f:
            self.encoded_actions = pickle.load(f)
        self.decoded_actions = {i: a for a, i in self.encoded_actions.items()}

    def getInitBoard(self) -> np.ndarray:
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        hands = np.zeros((3, 14), dtype=int)
        deck = []
        for i in range(54):
            deck.append(i // 4)
        shuffle(deck)

        for i in range(51):
            hands[i // 17, deck[i]] += 1
        for i in range(51, 54):
            hands[0, deck[i]] += 1

        last_move = np.zeros((14,))
        hands = hands.flatten()
        return np.concatenate((hands, last_move, [0, 0]))

    def getActionSize(self) -> int:
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 8542

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player
            action: action taken by current player
            passes: the number of passes

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        new_board = np.array(board)
        new_board[42:56] = 0
        action = self.decoded_actions[action]
        
        for card in action:
            new_board[card + 14 * player] -= 1
            new_board[card + 42] += 1

        if not action:
            new_board[56] += 1
        else:
            new_board[56] = 0
            
        if new_board[56] == 2:
            new_board[56] = 0
        
        new_board[57] = (new_board[57] + 1) % 3

        return new_board, (player + 1) % 3


    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        hand = np.array([board[:14]])
        last_move = []
        for i in np.argsort(-board[42:]):
            for _ in range(int(board[42:][i])):
                last_move.append(i)
        last_move = Play(last_move) if last_move else None

        game = GameState(hands=hand, last_move=last_move)
        valid_actions = [self.encoded_actions[tuple(action)] for action in game.legal_actions()]
        one_hot = np.zeros(self.getActionSize())
        one_hot[valid_actions] = 1
        return one_hot

    def getGameEnded(self, board, canonical=False):
        """
        Input:
            board: current board
            player: current player

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        wins = [np.sum(board[:14]) == 0, np.sum(board[14:28]) == 0, np.sum(board[28:42]) == 0]
        if canonical:
            landlord = int((3 - board[57]) % 3)
        else:
            landlord = 0
        farmers = {0, 1, 2} - {landlord}

        if board[57] == landlord:
            if wins[landlord]:
                return 1
            for farmer in farmers:
                if wins[farmer]:
                    return -1
        else:
            for farmer in farmers:
                if wins[farmer]:
                    return -1
            if wins[landlord]:
                return -1

        return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        # copies board
        canonical_board = np.array(board)
        shifted_hands = np.roll(canonical_board[:42], -14 * player)
        canonical_board[:42] = shifted_hands
        return canonical_board

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(board)
