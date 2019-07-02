# coding: utf-8

from scipy import sparse
from os import path
from src import config
from src.config import artifact_stores
from src.utils import io
from src.utils.converters import edge_list_to_adj_matrix


def main():
    node_labels = io.read_node_labels(config.train_node_labels_file)
    n_nodes = node_labels["node"].size

    container = artifact_stores.adjacency_matrices.multi_layer.create_artifact_container()
    container.save_params(
        source_files=[path.basename(f) for f in config.edge_lists_files],
        merged_layers=False
    )

    for index, file in enumerate(config.edge_lists_files):
        edge_list_df = io.read_edge_list(file)

        adj_matrix = edge_list_to_adj_matrix(
            src=edge_list_df["src"].values,
            dst=edge_list_df["dst"].values,
            weights=edge_list_df["weight"].values,
            dim=n_nodes
        )

        outfile = container.create_artifact_filepath(f"adjacency_matrix_{index}.npz")
        sparse.save_npz(file=outfile, matrix=adj_matrix)


if __name__ == "__main__":
    main()
