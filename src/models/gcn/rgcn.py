import torch
from torch import nn
from torch.nn import functional as F
from dgl import function as fn
from src.models.gcn.base import BaseGCN
from src.models.gcn.base import BaseGCNLayer


class RGCN(BaseGCN):
    """
    Relational Graph Convolutional Network classifier.

    References
    ----------
    M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling,
    “Modeling Relational Data with Graph Convolutional Networks,” arXiv:1703.06103 [cs, stat], Mar. 2017.
    """

    def __init__(
        self,
        graph,
        n_hidden_feats,
        n_hidden_layers,
        dropout,
        n_rels,
        n_bases,
        features,
        epochs,
        lr,
        weight_decay
    ):
        super().__init__(
            graph,
            n_hidden_feats,
            n_hidden_layers,
            F.relu,
            dropout,
            RGCNLayer,
            features,
            epochs,
            lr,
            weight_decay,
            n_rels=n_rels,
            n_bases=n_bases,
            self_loop=True,
        )


class RGCNLayer(BaseGCNLayer):
    def __init__(
        self,
        graph,
        n_in_feats,
        n_out_feats,
        activation,
        dropout,
        bias,
        is_input_layer,
        n_rels,
        n_bases,
        self_loop,
    ):
        nn.Module.__init__(self)
        self.graph = graph

        self.n_in_feats = n_in_feats
        self.n_out_feats = n_out_feats
        self.activation = activation
        self.self_loop = self_loop

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_out_feats))
        else:
            self.bias = None

        self.is_input_layer = is_input_layer
        self.n_rels = n_rels
        self.n_bases = n_bases

        if self.n_bases <= 0 or self.n_bases > self.n_rels:
            self.n_bases = self.n_rels

        self.loop_weight = nn.Parameter(torch.Tensor(n_in_feats, n_out_feats))

        # Add basis weights
        self.weight = nn.Parameter(
            torch.Tensor(self.n_bases, self.n_in_feats, self.n_out_feats)
        )

        if self.n_bases < self.n_rels:
            # Linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.n_rels, self.n_bases))

        self.reset_parameters()

    def reset_parameters(self):
        if self.self_loop:
            nn.init.xavier_uniform_(self.loop_weight)

        nn.init.xavier_uniform_(self.weight)

        if self.n_bases < self.n_rels:
            nn.init.xavier_uniform_(self.w_comp)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def propagate(self, h):
        if self.n_bases < self.n_rels:
            # Generate all weights from bases
            weight = self.weight.view(self.n_bases, self.n_in_feats * self.n_out_feats)
            weight = torch.matmul(self.w_comp, weight).view(
                self.n_rels, self.n_in_feats, self.n_out_feats
            )
        else:
            weight = self.weight

        if self.is_input_layer and h is None:

            def msg_func(edges):
                # For the input layer, matrix multiplication can be converted to an embedding lookup
                embed = weight.view(-1, self.n_out_feats)
                index = edges.data["type"] * self.n_in_feats + edges.src["id"]
                msg = embed.index_select(0, index) * edges.data["weight"]
                return {"msg": msg}

        else:

            def msg_func(edges):
                w = weight.index_select(dim=0, index=edges.data["type"])
                msg = torch.bmm(edges.src["h"].unsqueeze(1), w).squeeze()
                msg = msg * edges.data["weight"]
                return {"msg": msg}

        self.graph.update_all(msg_func, fn.sum(msg="msg", out="h"))

    def forward(self, h):
        if self.self_loop:
            if self.is_input_layer and h is None:
                loop_message = self.loop_weight
            else:
                loop_message = torch.mm(h, self.loop_weight)

            if self.dropout:
                loop_message = self.dropout(loop_message)

        if not (self.is_input_layer and h is None):
            self.graph.ndata["h"] = h

        # Send messages through all edges and update all nodes
        self.propagate(h)

        h = self.graph.ndata.pop("h")

        if self.self_loop:
            h = h + loop_message

        if self.bias is not None:
            h = h + self.bias

        if self.activation:
            h = self.activation(h)

        return h
