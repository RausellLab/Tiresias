import torch
import dgl
from scipy import sparse
from src.utils import matrix
from src.utils import converters
from src.utils import io


def get_device(use_cuda):
    return torch.device("cuda") if use_cuda else torch.device("cpu")


def load_labels(labels_file, use_cuda=False):
    """Loads labels for an experiment."""
    labels = io.read_node_labels(labels_file)["label"].values
    device = get_device(use_cuda)

    return torch.tensor(labels, dtype=torch.long, device=device)


def load_adj_matrices(
    adj_matrix_files, self_loop=False, normalization=None, use_cuda=False
):
    """Loads adjacency matrix for an experiment."""
    adj_matrices = []
    device = get_device(use_cuda)

    for adj_matrix_file in adj_matrix_files:
        adj_matrix = sparse.load_npz(adj_matrix_file).toarray()
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float, device=device)

        # Add self-loops
        if self_loop:
            matrix.add_self_loops(adj_matrix)
        else:
            matrix.remove_self_loops(adj_matrix)

        # Normalize the adjacency matrix
        if normalization == "sym":
            adj_matrix = matrix.sym_normalize(adj_matrix)
        elif normalization == "rw":
            adj_matrix = matrix.rw_normalize(adj_matrix)

        adj_matrices.append(adj_matrix)

    return adj_matrices


def load_graph(
    adj_matrix_files,
    n_nodes,
    self_loop=False,
    normalization=None,
    add_edge_type=False,
    add_node_ids=False,
    use_cuda=False,
):
    """Creates a DGL graph from a list of adjacency matrix files. Each
    adjacency matrix represents a layer of the graph.

    adj_matrix_files: list
        The list of files containing the adjacency matrix.

    n_nodes: int
        The number of nodes in the graph.

    self-loop: bool
        Whether self-loops are added to each adjacency matrix.

    normalization: str
        The type of normalization applied on each adjacency matrix.
            "sym": symmetrical normalization D⁻1/2 * A * D⁻1/2.
            "rw": random walk normalization D⁻1 * A.
            None: No normalization is applied.

    add_edge_type: bool
        Whether the type of the edge is added as edata.

    add_node_ids: bool
        Whether the id of each node is added as ndata.

    use_cuda: bool
        Whether the tensors are created using CUDA.
    """
    graph = dgl.DGLGraph()
    graph.add_nodes(n_nodes)

    device = get_device(use_cuda)

    for index, file in enumerate(adj_matrix_files):
        # Read matrix from files
        adj_matrix = torch.from_numpy(sparse.load_npz(file).toarray())

        # Add self-loops
        if self_loop:
            matrix.add_self_loops(adj_matrix)
        else:
            matrix.remove_self_loops(adj_matrix)

        # Normalize the adjacency matrix
        if normalization == "sym":
            adj_matrix = matrix.sym_normalize(adj_matrix)
        elif normalization == "rw":
            adj_matrix = matrix.rw_normalize(adj_matrix)

        # Get edge list from matrix
        src, dst, weights = converters.adj_matrix_to_edge_list(adj_matrix.numpy())
        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst)
        weights = torch.tensor(weights, dtype=torch.float, device=device)
        n_edges = src.size(0)

        edge_data = {"weight": weights.unsqueeze(1)}

        # Add edge type
        if add_edge_type:
            # todo: replace with new_full
            edge_data["type"] = torch.zeros(
                n_edges, dtype=torch.long, device=device
            ).fill_(index)

        # Add edges
        graph.add_edges(src, dst, edge_data)

        # Add node ids
        if add_node_ids:
            graph.ndata["id"] = torch.arange(n_nodes, dtype=torch.long, device=device)

    graph.set_n_initializer(dgl.init.zero_initializer)
    graph.set_e_initializer(dgl.init.zero_initializer)

    return graph


def load_embeddings(embeddings_file, use_cuda=False):
    """Load vector embeddings."""
    df = io.read_embeddings(embeddings_file)
    embeddings = df["embedding"].values.tolist()

    device = get_device(use_cuda)
    return torch.tensor(embeddings, dtype=torch.float, device=device)


def load_node_features(node_features_file, use_cuda=False):
    device = get_device(use_cuda)

    return torch.tensor(
        io.read_node_attributes(node_features_file).values,
        dtype=torch.float,
        device=device,
    )
