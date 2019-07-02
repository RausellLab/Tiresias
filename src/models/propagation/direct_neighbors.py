# coding: utf-8

import torch
from dgl import function as fn


class DirectNeighbors:
    """
    Naive network propagation method for positive-unlabelled learning on an undirected network.
    The model predicts that all the direct neighbors of a positively labelled node have a positive label.

    Parameters
    ----------
    graph: dgl.DGLGraph
        The graph on which the model is applied.
    """

    def __init__(self, graph):
        self.graph = graph
        self.predictions = None

    def reset_parameters(self):
        self.predictions = None

    def fit(self, train_labels, train_mask):
        """
        Trains the model.

        Parameters
        ----------
        train_labels: torch.LongTensor
            Tensor of target data of size n_train_nodes.

        train_mask: torch.ByteTensor
            Boolean mask of size n_nodes indicating the nodes used in training.
        """
        # Add initial node labels
        if train_labels.is_cuda:
            init_labels = torch.cuda.FloatTensor(self.graph.number_of_nodes()).fill_(0)
        else:
            init_labels = torch.zeros(self.graph.number_of_nodes(), dtype=torch.float)
        init_labels[train_mask] = train_labels.float()
        self.graph.ndata["l"] = init_labels

        # Propagate
        self.graph.update_all(
            message_func=fn.copy_src(src="l", out="m"),
            reduce_func=fn.max(msg="m", out="l"),
        )

        # Put back positive seed nodes
        self.graph.ndata["l"] = torch.max(self.graph.ndata["l"], init_labels)

        self.predictions = self.graph.ndata["l"]

    def predict_proba(self):
        """
        For each data sample, outputs the probability that it belongs to the positive class.

        Returns
        -------
        probabilities: torch.FloatTensor
            A 1-D tensor in which the i-th component holds the probability that the i-th data sample belongs to the
            positive class.
        """
        return self.predictions
