import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Supervised(nn.Module):
    def __init__(self, game):
        super(Supervised, self).__init__()

        self.action_size = 8542
        self.linear1 = nn.Linear(32, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, self.action_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return self.sigmoid(x)

    def play(self, x, possible_move_indices):
        temp = self.forward(x)
        max = -99999
        max_arg = 8541
        curr_sum = 0
        for index in possible_move_indices:
            curr_sum += temp[index]
            if index == 8541:
                continue
            if temp[index] > max:
                max = temp[index]
                max_arg = index
        #print(curr_sum, torch.sum(temp))
        #print("maxarg is ", max_arg)
        # if max_arg != 8541:
        #     for i in range(10):
        #         print(max_arg)
        return max_arg
        #return torch.argmax(self.forward(x))
