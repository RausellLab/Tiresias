import torch
import mlflow
from src.utils import matrix


class RwrM:
    """
    Random walk with restart on multi-layer networks.

    Parameters
    ----------
    adjacency_matrices: torch.FloatTensor of size n_layers * n_nodes * n_nodes
        Tensor containing the adjacency matrices of the multilayer graph on which the model will be applied.
        Each adjacency matrix corresponds to a layer of the graph.

    r: float
        Restart parameter.

    delta: float
        Probability for staying in a layer or jumping between the layers.

    max_iter: int
        Maximum number of iterations allowed.

    tol: float
        Convergence tolerance: threshold to consider the system at steady state.

    References
    ----------
    Valdeolivas, A., Tichit, L., Navarro, C., Perrin, S., Odelin, G., Levy, N., ... & Baudot, A. (2018).
    Random walk with restart on multiplex and heterogeneous biological networks. Bioinformatics, 35(3), 497-505.
    """
    def __init__(self, adjacency_matrices, r, delta=0.5, max_iter=1000, tol=1e-3):
        self.r = r
        self.max_iter = max_iter
        self.tol = tol

        self.device = adjacency_matrices.device
        self.n_layers = adjacency_matrices.size(0)
        self.n_nodes = adjacency_matrices.size(1)
        self.transition_matrix = build_transition_matrix(adjacency_matrices, delta)
        self.pred = None

    def reset_parameters(self):
        self.pred = None

    def fit(self, train_labels, train_mask):
        """
        Fits a positive-unlabelled learning network propagation model.

        Parameters
        ----------
        train_labels: torch.LongTensor = [n_train_nodes]
            Tensor of target (label) data for training.

        train_mask: torch.ByteTensor = [n_nodes]
            Boolean mask indicating the data instances that must be used for training and that are present in the
            train_labels tensor.
        """
        # Create labels tensor of size n_nodes
        labels = torch.zeros(self.n_nodes, dtype=torch.float, device=self.device)
        labels[train_mask] = train_labels.float()

        # Create initial predictions vector of size n_nodes * n_layers
        init_pred = torch.cat([labels / self.n_layers for _ in range(self.n_layers)])

        self.pred = init_pred.clone()
        prev_pred = torch.zeros(self.n_nodes * self.n_layers, dtype=torch.float, device=self.device)

        for i in range(self.max_iter):
            # Stop iterations if the system is considered at a steady state
            variation = torch.abs(self.pred - prev_pred).sum().item()
            mlflow.log_metric("variation", variation)

            if variation < self.tol:
                print(f"The method stopped after {i} iterations, variation={variation:.4f}.")
                break

            prev_pred = self.pred

            self.pred = (1 - self.r) * torch.matmul(self.transition_matrix, self.pred) + self.r * init_pred

        # Reshape tensor to
        self.pred = self.pred.view(self.n_layers, self.n_nodes)

        # Compute the global score for every node as the geometric mean of its n_layers proximity measure
        self.pred = torch.prod(self.pred, dim=0)
        self.pred = torch.pow(self.pred, 1 / self.n_layers)

    def predict_proba(self):
        """
        For each data sample, outputs the probability that it belongs to the positive class.

        Returns
        -------
        probabilities: torch.FloatTensor
            A 1-D tensor in which the i-th component holds the probability that the i-th data sample belongs to the
            positive class.
        """
        return self.pred


def build_transition_matrix(adjacency_matrices, delta):
    device = adjacency_matrices.device
    n_layers = adjacency_matrices.size(0)
    n_nodes = adjacency_matrices.size(1)

    # Create empty adjacency matrix for the multilayer graph
    dim = n_nodes * n_layers

    transition_matrix = torch.empty((dim, dim), dtype=torch.float, device=device)

    # Fill the adjacency matrix
    norm_I = delta / (n_layers - 1) * torch.eye(n_nodes, dtype=torch.float, device=device)

    for row in range(n_layers):
        start_row = row * n_nodes
        end_row = (row + 1) * n_nodes

        for col in range(n_layers):
            start_col = col * n_nodes
            end_col = (col + 1) * n_nodes

            if row == col:
                transition_matrix[start_row:end_row, start_col: end_col] = (1 - delta) * adjacency_matrices[row]
            else:
                transition_matrix[start_row:end_row, start_col: end_col] = norm_I

    transition_matrix = matrix.rw_normalize(transition_matrix)

    return transition_matrix
