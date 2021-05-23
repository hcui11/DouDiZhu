import numpy as np
from doudizhu import Game, Play
from mcts import MonteCarloTreeSearchNode
from pg import PGAgent
import torch
from tqdm import trange
import random
from greedy import NaiveGreedy, RandomPlayer
from visdom import Visdom

#random.seed(0)

class GameTransition():
    def __init__(self, current_state, prob, reward, next_state):
        #self.current_state = current_state
        #self.action = action
        self.prob = prob
        self.reward = reward
        self.next_state = next_state

def generate_transitions(agent):

    win_reward = 0
    no_reward = 0
    lose_reward = -1


    num_of_games = 1
    for _ in range(num_of_games):
        game = Game()

        game_transitions = []
        while game.get_winner() == -1:
            player = game.turn
            hands = game.hands[player]
            last_move = game.last_move
            last_deal = [] if last_move is None else last_move.cards
            possible_moves = game.legal_actions()
            played_cards = game.played_cards
            # last_move_player = game.last_move_player
            # last_move_player = [int(last_move_player == i) for i in range(3)]
            is_last_deal_landlord = int(game.last_move_player == 0)
            is_landlord = int(game.turn == 0)

            agent.current_state(hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord)
            current_state, action, score = agent.deal()

            play = Play(action)
            game.move(play)
            if game.get_winner() == -1:
                gt = GameTransition(current_state, score, no_reward, None)
            else:
                # if landlord wins
                if game.turn == 0:
                    game_transitions[-1].reward = lose_reward
                    game_transitions[-2].reward = lose_reward
                # if farmer 1 wins
                elif game.turn == 1:
                    game_transitions[-1].reward = lose_reward
                    game_transitions[-2].reward = win_reward
                # if farmer 2 wins
                else:
                    game_transitions[-1].reward = win_reward
                    game_transitions[-2].reward = lose_reward
                gt = GameTransition(current_state, score, win_reward, None)
            game_transitions.append(gt)
        # print(f"Player {game.get_winner()} wins!")
    return game_transitions

def get_loss(game_transitions):
    gamma = 0.99
    res = []
    sum_r = 0.
    for gt in reversed(game_transitions):
        sum_r *= gamma
        sum_r += gt.reward
        res.append(sum_r)
    res = list(reversed(res))
    loss = torch.zeros(1)
    for i, gt in enumerate(game_transitions):
        loss -= res[i] * torch.log(gt.prob)
    return loss



def learning_pool(agent, epochs):
    tr = trange(epochs, desc="loss")
    agent.model.train()
    for epoch in tr:
        game_transitions = generate_transitions(agent)

        # train player 0
        loss = get_loss(game_transitions[::3])
        loss += get_loss(game_transitions[1::3])
        loss += get_loss(game_transitions[2::3])
        loss = loss.sum()
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        tr.set_description("loss = %.5f"%loss.item())

def start_game(players, info=False):
    game = Game()
    while game.get_winner() == -1:
        player = game.turn
        hands = game.hands[player]
        last_move = game.last_move
        last_deal = [] if last_move is None else last_move.cards
        possible_moves = game.legal_actions()
        played_cards = game.played_cards
        # last_move_player = game.last_move_player
        # last_move_player = [int(last_move == i) for i in range(3)]
        is_last_deal_landlord = int(game.last_move == 0)
        is_landlord = int(game.turn == 0)

        players[player].current_state(hands, last_deal, possible_moves, played_cards, is_landlord, is_last_deal_landlord)
        action = players[player].play()
        #action = players[player].play(game.legal_actions(), player, game.hands[player], last_deal)
        play = Play(action)
        if info:
            print(f'player {game.turn}:', action)
        game.move(play)
    if info:
        print(f"Player {game.get_winner()} wins!")
    return game.get_winner()

def pg_vs_mcts(pg_agent):
    game = Game()
    state1 = Game(hands=game.hands+0)
    state2 = Game(hands=game.hands+0)
    mcts_agent1 = MonteCarloTreeSearchNode(state1, 1)
    mcts_agent2 = MonteCarloTreeSearchNode(state2, 2)
    mcts = [mcts_agent1, mcts_agent2]
    print('Game Start')
    while game.get_winner() == -1:
        if game.turn == 0:

            player = game.turn
            hands = game.hands[player]
            last_move = game.last_move
            last_deal = [] if last_move is None else last_move.cards
            possible_moves = game.legal_actions()

            agent.current_state(hands, last_deal, possible_moves)
            action = agent.deal_no_grad()
            play = Play(action)
            print('player 0:', action)
            game.move(play)

            mcts[0] = landlordAI_move(mcts[0], game, action)
            mcts[1] = landlordAI_move(mcts[1], game, action)
        else:
            mcts_agent = mcts[game.turn - 1]
            mcts_agent = mcts_agent.best_action()
            mcts_agent.parent = None
            move = mcts_agent.parent_action
            mcts[game.turn-1] = mcts_agent
            mcts[0] = landlordAI_move(mcts[0], game, move)
            mcts[1] = landlordAI_move(mcts[1], game, move)
            print(f'player {game.turn}:', move)
            game.move(move)
    print(f"Player {game.get_winner()} wins!")

def landlordAI_move(landlordAI, game, move):
    try:
        landlordAI = landlordAI.children[move]
        landlordAI.parent = None
    except:
        state = Game(hands=game.hands+0,
                     last_move=game.last_move,
                     turn=game.turn,
                     passes=game.passes)
        landlordAI = MonteCarloTreeSearchNode(state, 0)
    return landlordAI

def main(agent):
    game = Game()

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
            if game.turn == 0:
                player = game.turn
                hands = game.hands[player]
                last_move = game.last_move
                last_deal = [] if last_move is None else last_move.cards
                possible_moves = game.legal_actions()

                agent.current_state(hands, last_deal, possible_moves)
                action = agent.deal_no_grad()
                print("Agent played", action)
                input("Press anything to continue")
                play = Play(action)
                game.move(play)
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
                break

        print("\n\n")
    print(f"Player {game.get_winner()} wins!")

def save_model(model, path):
    model_saving_path = path
    print('saving model to ' + model_saving_path)
    torch.save(model.state_dict(), model_saving_path)

def load_model(model, path):
    model_loading_path = path
    print('loading model from ' + model_loading_path)
    model.load_state_dict(torch.load(model_loading_path))
    model.train()



if __name__ == '__main__':

    vis = Visdom()

    agent = PGAgent(learning_rate=0.01, device='cpu')
    #load_model(agent.model, "PG_param.pth")
    p0 = NaiveGreedy()
    p1 = NaiveGreedy()
    p2 = NaiveGreedy()
    # p0 = RandomPlayer()
    # p1 = RandomPlayer()
    # p2 = RandomPlayer()
    epochs = 10
    epoch_per_eval = 100

    win_ratio_ls = []
    epoch_ls = []
    for i in range(epochs//epoch_per_eval):

        learning_pool(agent, epoch_per_eval)
        #main(agent)

        players = [agent, p1, p2]
        #players = [p1, agent, agent]
        #players = [p0, p1, p2]
        #pg_vs_mcts(agent)

        total_game = 100
        counter = np.array([0, 0, 0])
        for _ in range(total_game):
            counter[start_game(players)] += 1
        counter = counter / total_game
        #print(counter)

        epoch_ls.append((i + 1) * epoch_per_eval)
        win_ratio_ls.append(counter[0])
        vis.line(X=epoch_ls, Y=win_ratio_ls, win='learning curve')
    save_model(agent.model, "PG_param.pth")

    # %%
    players = [agent, p1, p2]
    #players = [p1, agent, agent]
    #players = [p0, p1, p2]
    #pg_vs_mcts(agent)

    total_game = 1000
    counter = np.array([0, 0, 0])
    for _ in range(total_game):
        counter[start_game(players)] += 1
    counter = counter / total_game
    print(counter)
