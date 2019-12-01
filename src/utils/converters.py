from scipy import sparse


def adj_matrix_to_edge_list(A):
    """Converts an adjacency matrix to an edge list.

    Parameters
    ----------
    A: ndarray
        The input matrix.

    Returns
    -------
    src: ndarray
        A numpy array holding the source nodes.
    dst: ndarray
        A numpy array holding the destination nodes.
    weights: ndarray
        A numpy array holding the weights of the edges.
    """
    sp_matrix = sparse.coo_matrix(A)
    src = sp_matrix.row
    dst = sp_matrix.col
    weights = sp_matrix.data

    return src, dst, weights


def edge_list_to_adj_matrix(src, dst, weights, dim, dense=False):
    """Converts an edge list to an adjacency matrix.

    Parameters
    ----------
    src: ndarray
        A numpy array holding the source nodes.
    dst: ndarray
        A numpy array holding the destination nodes.
    weight: ndarray
        A numpy array holding the weights of the edges.
    dim: int
        The dimensions of the adjacency matrix (number of nodes in the graph).
    dense: bool
        Output a dense matrix (ndarray).

    Returns
    -------
    adj_matrix: ndarray or scipy.sparse
        The corresponding adjacency matrix.
    """
    adj_matrix = sparse.coo_matrix((weights, (src, dst)), shape=(dim, dim))

    if dense:
        return adj_matrix.toarray()

    return adj_matrix
