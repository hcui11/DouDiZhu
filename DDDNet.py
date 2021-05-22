import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class DDDNet(nn.Module):
    def __init__(self, game, args):
        super(DDDNet, self).__init__()
        # game params
        self.action_size = game.getActionSize()
        self.args = args

        self.linear1 = nn.Linear(58, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 1024)
        self.linear4 = nn.Linear(1024, 2048)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(2048)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.unsqueeze(0)
        s = F.relu(self.bn1(self.linear1(s)))                          # batch_size x 256
        s = F.relu(self.bn2(self.linear2(s)))                          # batch_size x 512
        s = F.relu(self.bn3(self.linear3(s)))                          # batch_size x 1024
        s = F.relu(self.bn4(self.linear4(s)))                          # batch_size x 2048

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
