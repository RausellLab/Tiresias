from abc import ABC, abstractmethod
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import mlflow


class BaseGCN(nn.Module, ABC):
    """Base model for GCN and RGCN.

    Parameters
    ----------
    graph: dgl.DGLGraph
        The graph on which the model is applied.

    n_hidden_feats: int
        The number of features for the input and hidden layers.

    n_hidden_layers: int
        The number of hidden layers.

    activation: torch.nn.functional
        The activation function used by the input and hidden layers.

    dropout: float
        The dropout rate.

    conv_layer: BaseGCNLayer
        The graph convolution layer of the model.

    features: torch.FloatTensor
        Feature matrix of size n_nodes * n_in_feats.

    epochs: int
        Number of epochs.

    lr: float
            Learning rate.

    weight_decay: float
        Weight decay (L2 penalty).
    """

    def __init__(
        self,
        graph,
        n_hidden_feats,
        n_hidden_layers,
        activation,
        dropout,
        conv_layer,
        features,
        epochs,
        lr,
        weight_decay,
        **kwargs,
    ):
        super().__init__()
        self.features = features
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        n_in_feats = (
            features.size(1) if features is not None else graph.number_of_nodes()
        )
        n_classes = 2

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            conv_layer(
                graph,
                n_in_feats,
                n_hidden_feats,
                activation,
                dropout=0,
                bias=True,
                is_input_layer=True,
                **kwargs,
            )
        )

        # Hidden layers
        for _ in range(n_hidden_layers):
            self.layers.append(
                conv_layer(
                    graph,
                    n_hidden_feats,
                    n_hidden_feats,
                    activation,
                    dropout,
                    bias=True,
                    is_input_layer=False,
                    **kwargs,
                )
            )

        # Output layer
        self.layers.append(
            conv_layer(
                graph,
                n_hidden_feats,
                n_classes,
                activation=None,
                dropout=dropout,
                bias=True,
                is_input_layer=False,
                **kwargs,
            )
        )

    def forward(self, x):
        """Defines how the model is run, from input to output.

        Parameters
        ----------
        x: torch.FloatTensor
            (Input) feature matrix of size n_nodes * n_in_feats.

        Return
        ------
        h: torch.FloatTensor
            Output matrix of size n_nodes * n_classes.
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def reset_parameters(self):
        """Reset the parameters (weights) of the layers of the model."""
        for layer in self.layers:
            layer.reset_parameters()

    def fit(self, train_labels, train_mask):
        """Trains the model.

        Parameters
        ----------

        train_labels: torch.LongTensor
            Tensor of target data of size n_train_nodes.

        train_mask: torch.ByteTensor
            Boolean mask of size n_nodes indicating the nodes used in training.
        """
        loss_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.train()
        for epoch in range(1, self.epochs + 1):
            # Forward
            optimizer.zero_grad()
            logits = self(self.features)
            loss = loss_criterion(logits[train_mask], train_labels)

            # Backward
            loss.backward()
            optimizer.step()

            mlflow.log_metric("loss", loss.item())
            print(f"Epoch: {epoch}/{self.epochs} | Loss {loss.item():.5f}")

    def predict_proba(self):
        """Outputs probabilities.

        Parameters
        ----------
        features: torch.FloatTensor
            Feature matrix of size n_nodes * n_in_feats.

        Returns
        -------
        probabilities: torch.FloatTensor
            A tensor of size n_nodes holding the probability for each node to belong to the positive class.
        """
        self.eval()
        with torch.no_grad():
            return F.softmax(self(self.features), dim=1)[:, 1]


class BaseGCNLayer(nn.Module, ABC):
    """Base class for GCNLayer and RGCNLayer.

    Parameters
    ----------
    graph: dgl.DGLGraph
        The graph on which the layer is applied.

    n_in_feats: int
        The number of input features of the layer.

    n_out_feats: int
        The number of output features of the layer.

    activation: torch.nn.functional
        The activation function applied.

    dropout: float
        The dropout rate.

    bias: bool
        Whether a bias is used.
    """

    @abstractmethod
    def __init__(self, graph, n_in_feats, n_out_feats, activation, dropout, bias):
        raise NotImplementedError("Subclasses must override __init__!")

    @abstractmethod
    def reset_parameters(self):
        raise NotImplementedError("Subclasses must override reset_parameters!")
