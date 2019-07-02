# coding: utf-8

from scipy import sparse
from os import path
from src import config
from src.config import artifact_stores
from src.utils.preprocess import merge_edge_lists
from src.utils import io
from src.utils.converters import edge_list_to_adj_matrix

OUTFILE_NAME = "adjacency_matrix.npz"


def main():
    node_labels = io.read_node_labels(config.train_node_labels_file)
    n_nodes = node_labels["node"].size

    merged_edge_lists_df = merge_edge_lists(
        (io.read_edge_list(f) for f in config.edge_lists_files)
    )

    adj_matrix = edge_list_to_adj_matrix(
        src=merged_edge_lists_df["src"].values,
        dst=merged_edge_lists_df["dst"].values,
        weights=merged_edge_lists_df["weight"].values,
        dim=n_nodes
    )

    container = artifact_stores.adjacency_matrices.merged_layer.create_artifact_container()
    container.save_params(
        source_files=[path.basename(f) for f in config.edge_lists_files],
        merged_layers=True
    )
    outfile = container.create_artifact_filepath(OUTFILE_NAME)

    sparse.save_npz(file=outfile, matrix=adj_matrix)


if __name__ == "__main__":
    main()
