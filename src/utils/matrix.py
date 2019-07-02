# coding: utf-8

import torch


def set_diagonal(A, val):
    """
    Sets diagonal elements of a matrix.

    Parameters
    ----------
    A: torch.Tensor
        The matrix to modify.
    """

    diag_mask = torch.eye(A.size(0), dtype=torch.uint8)
    A[diag_mask] = val


def remove_self_loops(A):
    """
    Removes self loops in an adjacency matrix of a graph.

    Parameters
    ----------
    A: torch.Tensor
        The adjacency matrix of the graph.
    """
    set_diagonal(A, 0)


def add_self_loops(A):
    """
    Adds self loops in an adjacency matrix of a graph.

    Parameters
    ----------
    A: torch.Tensor
        The adjacency matrix of the graph.
    """
    set_diagonal(A, 1)


def sym_normalize(A):
    """
    Symmetrical normalization: computes D^-1/2 * A * D^-1/2.

    Parameters
    ----------
    A: torch.Tensor
        The matrix to normalize.

    Returns
    -------
    A_norm: torch.FloatTensor
        The normalized adjacency matrix.
    """
    degs = A.sum(dim=1)
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 1
    return A * norm[:, None] * norm[None, :]


def rw_normalize(A):
    """
    Random walk normalization: computes D^⁼1 * A.

    Parameters
    ----------
    A: torch.Tensor
        The matrix to normalize.

    Returns
    -------
    A_norm: torch.FloatTensor
        The normalized adjacency matrix.
    """
    degs = A.sum(dim=1)
    degs[degs == 0] = 1
    return A / degs[:, None]
