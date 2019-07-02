# coding: utf-8

import numpy as np


def cumulative_distribution_function(ranks, n_nodes):
    """
    Cumulative distribution function.

    Parameters
    ----------
    ranks: np.array
        The rank of each positive gene.

    n_nodes: int
        The total number of genes.

    Returns
    -------
    cumulative_distribution:
        The proportion of genes ranked in the top k genes as a function of k.
    """
    count = np.fromiter(
        (ranks[ranks <= i].size for i in range(1, n_nodes + 1)), dtype=int
    )
    return count / ranks.size
