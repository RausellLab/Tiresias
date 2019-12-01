import torch


def matrix_symmetry(matrix):
    """Checks the symmetry of the input matrix.

    Parameters
    ----------
    matrix: torch.FloatTensor
        The matrix to check.
    """
    if torch.eq(matrix, matrix.t()).all().item() == 0:
        raise ValueError("The input matrix is not symmetrical.")


def pu_labels(labels):
    """Checks that the input labels correspond to a PU learning setting.

    Parameters
    ----------
    labels: torch.LongTensor
        The label vector to check.
    """
    if not labels.dim() == 1:
        raise ValueError("Input tensor must have a dimension of 1.")

    if not ((labels == 0) | (labels == 1)).nonzero().size(0) == labels.size(0):
        raise ValueError("Input tensor contains value(s) different from 0 and 1.")


def adj_matrix_labels_dim(adj_matrix, labels):
    """Checks that the dimensions of the adjacency matrix and the labels vector
    are compatible.

    Parameters
    ----------
    adj_matrix: torch.FloatTensor
        The adjacency matrix to check.

    labels: torch.LongTensor
        The label vector to check.
    """
    if not adj_matrix.size() == torch.Size([labels.size(0), labels.size(0)]):
        raise ValueError(
            "The dimensions of the adjacency matrix and the label vector are incompatible."
        )


def features_labels_dim(features, labels):
    """Checks that the dimensions of the features matrix and the labels vector
    are compatible.

    Parameters
    ----------
    features: torch.FloatTensor
        The feature matrix to check.

    labels: torch.LongTensor
        The label vector to check.
    """
    if not features.size() == torch.Size([labels.size(0), labels.size(0)]):
        raise ValueError(
            "The dimensions of the adjacency matrix and the label vector are incompatible."
        )
