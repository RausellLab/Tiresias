from pecanpy import node2vec
from src.utils import io


def run(
    input_edge_list,
    output_file,
    walk_length=80,
    n_walks=10,
    epochs=1,
    p=0.5,
    q=1,
    verbose=True,
    directed_graph=False,
    weighted_graph=False,
    workers = 4 
):
    """Wrapper around node2vec pecanpy implementation.

    Parameters
    ----------
    input_edge_list: str
        Absolute path of the input edge list file.

    output_file: str
        Absolute path of the output file (embedding or random walk file).

    dimensions: int (default: 128)
        Number of dimensions of the generated vector embeddings.

    walk_length: int (default: 80)
        Length of walk per source node.

    n_walks: int (default: 10)
        Number of walks per source node.

    context_size: int (default: 10)
        Context size in Word2Vec.

    epochs: int (default: 1)
        Number of epochs in stochastic gradient descent.

    p: int (default: 1)
        Return hyperparameter.

    q: int (default: 1)
        Inout hyperparameter.

    verbose: bool (default: True)
        Verbosity of the output.

    directed_graph: bool (default: False)
        Indicates whether the graph is directed.

    weighted_graph: bool (default: False)
        Indicates whether the graph is weighted.

    output_random_walks: bool (default: False)
        Output random walks instead of node embeddings.

    References
    ----------
    Liu R, Krishnan A (2021) PecanPy: a fast, efficient, and parallelized Python implementation of node2vec. Bioinformatics https://doi.org/10.1093/bioinformatics/btab202    """
    
    g = node2vec.DenseOTF(p=p, q=q, workers=workers , verbose=verbose)
    g.read_edg(input_edge_list, weighted=weighted_graph, directed=directed_graph) # load graph from edgelist file
    random_walks_list = g.simulate_walks(num_walks=n_walks, walk_length=walk_length) # generate node2vec walks
    io.write_random_walks(random_walks_list, output_file)
    del(g)
    print("Done!")

