import torch
from torch import nn


class RegressorNet(nn.Module):
    """
    A class for solving regression problems.
    """

    def __init__(self, in_dim):
        super().__init__()

        self.half_dim = in_dim // 2
        self.dense_1_1 = nn.Linear(self.half_dim, 64)
        self.dense_1_2 = nn.Linear(self.half_dim, 64)
        self.dense_2 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x1 = x[:, : self.half_dim]
        x2 = x[:, self.half_dim :]
        h1 = nn.Tanh()(self.dense_1_1(x1))
        h2 = nn.ReLU()(self.dense_1_2(x1))
        h = torch.cat([h1, h2], dim=1)
        h = nn.ReLU()(self.dense_2(h))
        h = self.dropout(h)
        out = nn.ReLU()(self.out(h))
        return out
