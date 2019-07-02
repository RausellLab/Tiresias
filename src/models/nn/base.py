# coding: utf-8

from abc import ABC
from abc import abstractmethod
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import mlflow


class BaseNN(nn.Module, ABC):
    """
    Base class for classifiers such as Multilayer Perceptron and Logistic Regression.

    Parameters
    ----------
    features: torch.FloatTensor
        Tensor of input features for all data instances.

    epochs: int
        Number of epochs used to train the model.

    lr: float
        Learning rate used to train the model.
    """
    def __init__(self, features, epochs=1, lr=0.01):
        super().__init__()
        self.features = features
        self.epochs = epochs
        self.lr = lr

    @abstractmethod
    def reset_parameters(self):
        raise NotImplementedError("Subclasses must override reset_parameters!")

    def fit(self, train_labels, train_mask):
        """
        Trains the model.

        Parameters
        ----------
        train_labels: torch.LongTensor
            Tensor of train target (label) data.

        train_mask: torch.ByteTensor
            Boolean mask indicating the data instances that must be used for training.
        """
        loss_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        train_data = self.features[train_mask]

        self.train()
        for epoch in range(1, self.epochs + 1):
            # Forward
            optimizer.zero_grad()
            logits = self(train_data)
            loss = loss_criterion(logits, train_labels)

            # Backward
            loss.backward()
            optimizer.step()

            mlflow.log_metric("loss", loss.item())
            print(f"Epoch: {epoch}/{self.epochs} | Loss {loss.item():.5f}")

    def predict_proba(self):
        """
        For each data sample, outputs the probability that it belongs to the positive class.

        Returns
        -------
        pos_class_proba: torch.FloatTensor
            A 1-D tensor in which the i-th component holds the probability that the i-th data sample belong to the
            positive class.
        """
        self.eval()
        with torch.no_grad():
            proba = F.softmax(self(self.features), dim=1)
            pos_class_proba = proba[:, 1]
            return pos_class_proba
