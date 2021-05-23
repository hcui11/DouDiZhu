import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PG(nn.Module):
    def __init__(self):
        super(PG, self).__init__()
        self.linear1 = nn.Linear(58, 32)
        self.linear2 = nn.Linear(32, 1)


        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x = hand, last, cand_deal = 1x42
        hand: 1x14
        last: 1x14
        cand_deal: candidate deal, 1x14
        played_cards: 1x14
        Returns: 1x1
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

class PGAgent():
    def __init__(self, learning_rate, device='cpu'):
        self.device = device
        self.model = PG().to(self.device)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def current_state(self, hands, last_deal, legal_actions, played_cards, is_landlord, is_last_deal_landlord):
        """
        hands: 13
        last_deal: D
        legal_hands: NxM
        played_cards: 13
        is_landlord: 1
        is_last_deal_landlord: 1
        """
        self.hands = hands
        self.last_deal = last_deal
        self.legal_actions = legal_actions
        self.played_cards = played_cards
        self.is_landlord = is_landlord
        self.is_last_deal_landlord = is_last_deal_landlord

    @staticmethod
    def cards_to_list(cards):
        """
        action: M
        Returns: 14
        """
        rt = [0 for _ in range(14)]
        for i in cards:
            rt[i] += 1
        return rt

    @staticmethod
    def get_state(hands, last, action, played_cards, is_landlord, is_last_deal_landlord):
        hands = torch.FloatTensor([hands])
        last = torch.FloatTensor([last])
        action = torch.FloatTensor([action])
        played_cards = torch.FloatTensor([played_cards])
        is_landlord = torch.FloatTensor([[is_landlord]])
        is_last_deal_landlord = torch.FloatTensor([[is_last_deal_landlord]])
        return torch.cat([hands, last, action, played_cards, is_landlord, is_last_deal_landlord], dim=1)

    def deal(self):
        """
        call it after current_state
        Returns state, action, score
        """
        hands = self.hands
        last = self.cards_to_list(self.last_deal)
        scores = torch.zeros(len(self.legal_actions))
        played_cards = self.played_cards
        is_landlord = self.is_landlord
        is_last_deal_landlord = self.is_last_deal_landlord

        for i, action in enumerate(self.legal_actions):
            action = self.cards_to_list(action)
            game_state = self.get_state(hands, last, action, played_cards, is_landlord, is_last_deal_landlord).to(self.device)
            scores[i] = self.model(game_state)
        scores = torch.softmax(scores, dim=0)
        best_action = np.random.choice(np.arange(0, len(scores), 1), p=scores.cpu().detach().numpy())
        best_action = self.legal_actions[scores.argmax().item()]
        return game_state, best_action, scores.max()

    def deal_no_grad(self):
        self.model.eval()
        with torch.no_grad():
            _, action, _ = self.deal()
        return action

    def play(self):
        return self.deal_no_grad()