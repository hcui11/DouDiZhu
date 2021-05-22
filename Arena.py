import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, prev_model, new_model, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.landlord = prev_model
        self.farmers = new_model
        self.game = game
        self.display = display

    def playGame(self, verbose=False) -> int:
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if landlord, -1 if farmers)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        curPlayer = 0
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board) == 0:
            it += 1
            # if verbose:
            #     assert self.display
            #     print("Turn ", str(it), "Player ", str(curPlayer))
            #     self.display(board)
            player_func = self.landlord if curPlayer == 0 else self.farmers
            action = player_func(self.game.getCanonicalForm(board, curPlayer))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        # if verbose:
        #     assert self.display
        #     print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
        #     self.display(board)
        return 1 if curPlayer == 1 else -1

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and farmers starts
        num/2 games.

        Returns:
            oneWon: games won by landlord
            twoWon: games won by farmers
            draws:  games won by nobody
        """

        num = int(num / 2)
        prevWon = 0
        newWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                prevWon += 1
            elif gameResult == -1:
                newWon += 1
            else:
                draws += 1

        self.landlord, self.farmers = self.farmers, self.landlord

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                prevWon += 1
            elif gameResult == 1:
                newWon += 1
            else:
                draws += 1

        return prevWon, newWon, draws
