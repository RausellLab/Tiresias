# coding: utf-8

from torch import nn
from torch.nn import init
from src.models.nn.base import BaseNN


class LogisticRegression(BaseNN):
    """A simple logistic regression for binary classification."""

    def __init__(self, features, epochs=1, lr=0.01):
        super().__init__(features, epochs, lr)

        n_in_feats = features.size(1)
        n_classes = 2

        self.linear = nn.Linear(n_in_feats, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)
