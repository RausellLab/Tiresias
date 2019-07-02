import torch
from torch import nn
from torch.nn import functional as F
from dgl import function as fn
from torch.nn import init
from src.models.gcn.base import BaseGCN
from src.models.gcn.base import BaseGCNLayer


class GCN(BaseGCN):
    """
    Graph Convolutional Network classifier.

    References
    ----------
    T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,”
    arXiv:1609.02907 [cs, stat], Sep. 2016.
    """

    def __init__(
        self,
        graph,
        n_hidden_feats,
        n_hidden_layers,
        dropout,
        features,
        epochs,
        lr
    ):
        super().__init__(
            graph,
            n_hidden_feats,
            n_hidden_layers,
            F.relu,
            dropout,
            GCNLayer,
            features,
            epochs,
            lr,
            weight_decay=0
        )


class GCNLayer(BaseGCNLayer):
    def __init__(
        self, graph, n_in_feats, n_out_feats, activation, dropout, bias, is_input_layer
    ):
        nn.Module.__init__(self)
        self.graph = graph
        self.activation = activation

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_out_feats))
        else:
            self.bias = None

        self.is_input_layer = is_input_layer

        self.weight = nn.Parameter(torch.Tensor(n_in_feats, n_out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, h):
        if self.is_input_layer and h is None:
            h = self.weight
        else:
            h = torch.mm(h, self.weight)

        if self.dropout:
            h = self.dropout(h)

        self.graph.ndata["h"] = h

        # Send messages through all edges and update all nodes
        self.graph.update_all(
            fn.src_mul_edge(src="h", edge="weight", out="m"), fn.sum(msg="m", out="h")
        )

        h = self.graph.ndata.pop("h")

        if self.bias is not None:
            h = h + self.bias

        if self.activation:
            h = self.activation(h)

        return h
