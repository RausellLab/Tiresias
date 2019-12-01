import numpy as np
import pandas as pd


def read_node_labels(file):
    return pd.read_csv(
        file,
        header=None,
        names=["node", "label"],
        skiprows=1,
        sep="\t",
        dtype={"node": np.int64, "label": np.int64},
    )


def read_edge_list(file):
    return pd.read_csv(
        file,
        header=None,
        names=["src", "dst", "weight"],
        skiprows=1,
        sep="\t",
        dtype={"src": np.int64, "dst": np.int64, "weight": np.float64},
    )


def read_random_walks(random_walks_file):
    """Reads a list of random walks from a text file.

    Parameters
    ----------
    random_walks_file: str
        The path of the file containing the list of random walks.

    Returns
    -------
    random_walks: ndarray
        The list of the random walks as a numpy array with dtype=int32.
    """
    return np.loadtxt(random_walks_file, delimiter=" ", dtype=np.int32)


def read_embeddings(input_file):
    """Loads vector embeddings from text file.

    Parameters
    ----------
    input_file: str
        Absolute path of the text file containing the embeddings.

    Returns
    -------
    df: pandas.DataFrame
        A dataframe containing the embeddings.

    Input file example
    ------------------
    2 5
    14 0.043806 0.288255 0.0781747 0.105145 0.0478589
    34 0.0455626 -0.15585 -0.0748686 -0.141817 0781747
    """

    # Read header
    header = np.loadtxt(input_file, max_rows=1, dtype=np.int32)
    nb_nodes, dimension = header[0], header[1]

    # Read node numbers
    nodes = np.loadtxt(input_file, skiprows=1, usecols=0, dtype=int)

    if nb_nodes != nodes.size:
        raise RuntimeError(
            f"Incorrect number of nodes. Expected: {nb_nodes}. Actual: {nodes.size}."
        )

    # Read vector embeddings
    embeddings = np.loadtxt(input_file, skiprows=1, dtype=float)
    embeddings = embeddings[:, 1:]

    if dimension != embeddings.shape[1]:
        raise RuntimeError(
            f"Incorrect vector dimension. Expected: {dimension}. Actual: {embeddings.shape[1]}."
        )

    df = pd.DataFrame(data={"embedding": embeddings.tolist()}, index=nodes)
    df.index.name = "node"
    df.sort_index(inplace=True)

    return df


def read_node_attributes(file):
    return pd.read_csv(file, sep="\t", index_col=[0], dtype=float)
