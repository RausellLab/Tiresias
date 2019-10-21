import torch
import mlflow


class NetPropWithRestart:
    """
    Base class for propagation with restart on a network with positive and unlabelled data.

    Parameters
    ----------
    adjacency_matrix: torch.FloatTensor
        Normalized adjacency matrix of the graph on which the model will be applied.

    r: float
        Restart parameter.

    max_iter: int
        Maximum number of iterations allowed.

    tol: float
        Convergence tolerance: threshold to consider the system at steady state.
    """
    def __init__(self, adjacency_matrix, r, max_iter=1000, tol=1e-3):
        self.adjacency_matrix = adjacency_matrix
        self.predictions = None
        self.device = adjacency_matrix.device
        self.r = r
        self.max_iter = max_iter
        self.tol = tol

    def reset_parameters(self):
        self.predictions = None

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
        n_nodes = train_mask.size(0)
        labels = torch.zeros(n_nodes, dtype=torch.float, device=self.device)
        labels[train_mask] = train_labels.float()

        self.predictions = labels.clone()

        prev_predictions = torch.zeros(n_nodes, dtype=torch.float, device=self.device)

        for i in range(self.max_iter):
            # Stop iterations if the system is considered at a steady state
            variation = torch.abs(self.predictions - prev_predictions).sum().item()
            mlflow.log_metric("variation", variation)

            if variation < self.tol:
                print(f"The method stopped after {i} iterations, variation={variation:.4f}.")
                break

            prev_predictions = self.predictions

            self.predictions = (
                    self.r * torch.matmul(self.adjacency_matrix, self.predictions)
                    + (1 - self.r) * labels
            )

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
