# coding: utf-8

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from src.models.nn.base import BaseNN


class MLP(BaseNN):
    """A simple multilayer perceptron with a single hidden layer."""

    def __init__(self, features, n_hidden_feats, epochs=1, lr=0.01, dropout=0):
        super().__init__(features, epochs, lr)

        n_in_feats = features.size(1)
        n_classes = 2

        self.fc1 = nn.Linear(n_in_feats, n_hidden_feats)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(n_hidden_feats, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
