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
        """
        # self.hands = np.zeros((3, 14), dtype=int)
        # self.player = 0
        # self.last_move = None
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
        return np.concatenate((hands, last_move))

    def getActionSize(self) -> int:
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 8542

    def getNextState(self, board, player, action, passes):
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
        new_board = np.abs(board)
        new_board[42:] = 0
        action = self.decoded_actions[action]
        
        for card in action:
            new_board[card + 14 * player] -= 1
            new_board[card + 42] += 1

        if not action:
            passes += 1
        else:
            passes = 0
            
        if passes == 2:
            passes = 0

        return new_board, (player + 1) % 3, passes


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
        hand = np.abs([board[14 * player: 14 * (player + 1)]])
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

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        win0 = np.sum(board[:14]) == 0
        win1 = np.sum(board[14:28]) == 0
        win2 = np.sum(board[28:42]) == 0
        if player == 0:
            if win0:
                return 1
            elif win1 or win2:
                return -1
        else:
            if win1 or win2:
                return 1
            elif win0:
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
        canonical_board = np.abs(board)
        canonical_board[:42] *= -1
        canonical_board[14 * player: 14 * (player + 1)] *= -1
        return canonical_board

    # def getSymmetries(self, board, pi):
    #     """
    #     Input:
    #         board: current board
    #         pi: policy vector of size self.getActionSize()

    #     Returns:
    #         symmForms: a list of [(board,pi)] where each tuple is a symmetrical
    #                    form of the board and the corresponding pi vector. This
    #                    is used when training the neural network from examples.
    #     """
    #     pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(board)
