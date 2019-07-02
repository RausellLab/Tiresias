# coding: utf-8

import os
import time
import tempfile
import pandas as pd
import torch
import argparse
import psutil
from scipy import sparse
from src.utils import data_loaders
from src.utils import io
from src.models import DirectNeighbors
from src.models import LabelPropagation
from src.models import LabelSpreading
from src.models import GCN
from src.models import RGCN
from src.models import LogisticRegression
from src.models import MLP
from src.features import node2vec_snap
from src.features import skip_gram
from src.config import (
    edge_lists_files,
    train_node_labels_file
)


def print_heading(text, level=2):
    underline = "".join(("-" for _ in range(len(text))))

    if level == 1:
        print(underline)

    print(text)
    print(f"{underline}")

    if level == 1:
        print()


def time_func(f):
    def func_timer(*args, **kwargs):
        start = time.time()
        f(*args, **kwargs)
        end = time.time()
        duration = end - start
        print(f"{f.__name__} took {duration:.3f} seconds")
        return duration

    return func_timer


@time_func
def rw_walk_multi(edge_list_filenames, rand_walks_filenames):
    for edge_list_filename, rand_walks_filename in zip(
        edge_list_filenames, rand_walks_filenames
    ):
        node2vec_snap.run(
            edge_list_filename,
            rand_walks_filename,
            walk_length=80,
            n_walks=10,
            p=1,
            q=1,
            verbose=True,
            directed_graph=True,
            weighted_graph=True,
            output_random_walks=True,
        )


def print_running_time(duration, n_epochs, n_bootstraps, n_loocv_iterations):
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"Estimated running time for {n_epochs} epochs, {n_bootstraps} bootstraps, "
        f"{n_loocv_iterations} LOOCV iterations:"
    )
    print(
        f"> {hours:0>2} hours, {minutes:0>2} minutes, {seconds:.3f} seconds",
        end="\n\n",
    )


def with_concat_networks(
    edge_list_files,
    labels,
    train_mask,
    n_nodes,
    n_pos_nodes,
    estimate_running_time,
    estimate_running_time_with_preprocessing_steps,
    use_cuda,
    n_cpus
):
    print_heading("Concatenated networks", level=1)

    # Remove duplicated edges and keep the highest weights
    concat_edge_list_df = pd.concat(
        (io.read_edge_list(f) for f in edge_list_files), ignore_index=True
    )

    concat_edge_list_df = concat_edge_list_df.sort_values(
        "weight", ascending=False
    ).drop_duplicates(["src", "dst"], keep="first")

    with tempfile.NamedTemporaryFile() as adj_matrix_tmp_file, tempfile.NamedTemporaryFile() as edge_list_tmp_file:
        # Generate adjacency matrix
        adj_matrix = sparse.coo_matrix(
            (
                concat_edge_list_df["weight"].values,
                (concat_edge_list_df["src"].values, concat_edge_list_df["dst"].values),
            ),
            shape=(n_nodes, n_nodes),
        )

        # Write adj matrix to file
        sparse.save_npz(file=adj_matrix_tmp_file, matrix=adj_matrix)

        # write SNAP compatible edge list
        concat_edge_list_df.to_csv(
            edge_list_tmp_file.name, sep=" ", header=False, index=False
        )

        graph = data_loaders.load_graph(
            [adj_matrix_tmp_file.name], n_nodes, use_cuda=use_cuda
        )
        adj_matrix = data_loaders.load_adj_matrices(
            [adj_matrix_tmp_file.name], use_cuda=use_cuda
        )[0]

        print("Number of nodes: ", n_nodes)
        print("Number of positive nodes: ", n_pos_nodes)
        print("Number of edges: ", graph.number_of_edges())
        print()

        print_heading("Direct Neighbors")
        direct_neighbors = DirectNeighbors(graph)
        direct_neighbors.fit = time_func(direct_neighbors.fit)
        duration = direct_neighbors.fit(
            train_labels=labels[train_mask], train_mask=train_mask
        )
        estimate_running_time(duration, n_epochs=1, n_bootstraps=1)

        print_heading("Label Propagation")
        label_propagation = LabelPropagation(adj_matrix=adj_matrix)
        label_propagation.fit = time_func(label_propagation.fit)
        duration = label_propagation.fit(
            train_labels=labels[train_mask], train_mask=train_mask
        )
        estimate_running_time(duration, n_epochs=1, n_bootstraps=1)

        print_heading("Label Spreading")
        label_spreading = LabelSpreading(adj_matrix=adj_matrix)
        label_spreading.fit = time_func(label_spreading.fit)
        duration = label_spreading.fit(
            train_labels=labels[train_mask], train_mask=train_mask
        )
        estimate_running_time(duration, n_epochs=1, n_bootstraps=1)

        print_heading("GCN")
        gcn = GCN(
            graph=graph,
            n_in_feats=None,
            n_hidden_feats=5,
            n_classes=2,
            n_hidden_layers=0,
            dropout=0,
        )

        if use_cuda:
            gcn.cuda()

        gcn.fit = time_func(gcn.fit)
        duration = gcn.fit(
            features=None,
            train_labels=labels[train_mask],
            train_mask=train_mask,
            epochs=1,
            lr=0.01,
            weight_decay=0,
        )
        estimate_running_time(duration)

        print_heading("RGCN")
        graph = data_loaders.load_graph(
            [adj_matrix_tmp_file.name],
            n_nodes,
            add_edge_type=True,
            add_node_ids=True,
            normalization="sym",
            use_cuda=use_cuda,
        )

        rgcn = RGCN(
            graph=graph,
            n_in_feats=None,
            n_hidden_feats=4,
            n_classes=2,
            n_hidden_layers=0,
            dropout=0,
            n_rels=1,
            n_bases=-1,
            self_loop=False,
        )

        if use_cuda:
            rgcn.cuda()

        rgcn.fit = time_func(rgcn.fit)
        duration = rgcn.fit(
            features=None,
            train_labels=labels[train_mask],
            train_mask=train_mask,
            epochs=1,
            lr=0.01,
            weight_decay=0,
        )
        estimate_running_time(duration)

        with tempfile.NamedTemporaryFile() as rw_tmp_file:
            print_heading("Random Walks")
            node2vec_snap.run = time_func(node2vec_snap.run)
            rw_duration = node2vec_snap.run(
                edge_list_tmp_file.name,
                rw_tmp_file.name,
                walk_length=80,
                n_walks=10,
                p=1,
                q=1,
                verbose=True,
                directed_graph=True,
                weighted_graph=True,
                output_random_walks=True,
            )
            print()

            with tempfile.NamedTemporaryFile() as sg_tmp_file:
                print_heading("Skip-Gram")
                skip_gram.run = time_func(skip_gram.run)
                sg_duration = skip_gram.run(
                    random_walk_files=[rw_tmp_file.name],
                    output_file=sg_tmp_file.name,
                    dimensions=128,
                    context_size=10,
                    epochs=100,
                    workers=n_cpus,
                )
                print()

                embeddings = data_loaders.load_embeddings(
                    embeddings_file=sg_tmp_file.name, use_cuda=use_cuda
                )

                print_heading("Logistic Regression")
                logistic_regression = LogisticRegression(n_in_feats=128, n_classes=2)

                if use_cuda:
                    logistic_regression.cuda()

                logistic_regression.fit = time_func(logistic_regression.fit)
                lr_duration = logistic_regression.fit(
                    train_labels=labels[train_mask],
                    train_mask=train_mask,
                    features=embeddings,
                    epochs=1,
                    lr=0.01,
                )
                estimate_running_time(lr_duration)

                print_heading("Random Walks + Skip-Gram + Logistic Regression")
                estimate_running_time_with_preprocessing_steps(
                    rw_duration + sg_duration,
                    lr_duration
                )

                print_heading("MLP")
                mlp = MLP(n_in_feats=128, n_hidden_feats=16, n_classes=2, dropout=0.1)

                if use_cuda:
                    mlp.cuda()

                mlp.fit = time_func(mlp.fit)
                mlp_duration = mlp.fit(
                    train_labels=labels[train_mask],
                    train_mask=train_mask,
                    features=embeddings,
                    epochs=1,
                    lr=0.01,
                )
                estimate_running_time(mlp_duration)

                print_heading("Random Walks + Skip-Gram + MLP")
                estimate_running_time_with_preprocessing_steps(
                    rw_duration + sg_duration,
                    mlp_duration
                )


def with_multilayer_network(
    edge_lists_files,
    labels,
    train_mask,
    n_nodes,
    n_pos_nodes,
    estimate_running_time,
    estimate_running_time_with_preprocessing_steps,
    use_cuda,
    n_cpus
):
    print_heading("Multilayer network", level=1)

    tmp_files = []

    try:
        adj_matrix_tmp_files = []
        edge_list_tmp_files = []

        for file in edge_lists_files:
            # Read edge list
            edge_list_df = io.read_edge_list(file)

            # Generate adjacency matrix
            adj_matrix = sparse.coo_matrix(
                (
                    edge_list_df["weight"].values,
                    (edge_list_df["src"].values, edge_list_df["dst"].values),
                ),
                shape=(n_nodes, n_nodes),
            )

            # Create temp file
            adj_matrix_tmp_file = tempfile.NamedTemporaryFile()
            adj_matrix_tmp_files.append(adj_matrix_tmp_file)

            # Write adj matrix to file
            sparse.save_npz(file=adj_matrix_tmp_file, matrix=adj_matrix)

            edge_list_tmp_file = tempfile.NamedTemporaryFile()

            # write SNAP compatible edge list
            edge_list_df.to_csv(
                edge_list_tmp_file.name, sep=" ", header=False, index=False
            )
            edge_list_tmp_files.append(edge_list_tmp_file)

        tmp_files.extend(adj_matrix_tmp_files)
        tmp_files.extend(edge_list_tmp_files)

        graph = data_loaders.load_graph(
            [f.name for f in adj_matrix_tmp_files],
            n_nodes,
            add_edge_type=True,
            add_node_ids=True,
            normalization="sym",
            use_cuda=use_cuda,
        )

        print("Number of nodes: ", n_nodes)
        print("Number of positive nodes: ", n_pos_nodes)
        print("Number of edges: ", graph.number_of_edges())
        print("Number of layers: ", len(edge_lists_files))
        print()

        print_heading("RGCN")
        rgcn = RGCN(
            graph=graph,
            n_in_feats=None,
            n_hidden_feats=4,
            n_classes=2,
            n_hidden_layers=0,
            dropout=0,
            n_rels=len(adj_matrix_tmp_files),
            n_bases=-1,
            self_loop=False,
        )

        if use_cuda:
            rgcn.cuda()

        rgcn.fit = time_func(rgcn.fit)
        duration = rgcn.fit(
            features=None,
            train_labels=labels[train_mask],
            train_mask=train_mask,
            epochs=1,
            lr=0.01,
            weight_decay=0,
        )
        estimate_running_time(duration)

        print_heading("Random Walks")
        rw_tmp_files = [tempfile.NamedTemporaryFile() for _ in edge_lists_files]

        tmp_files.extend(rw_tmp_files)

        rw_duration = rw_walk_multi(
            [f.name for f in edge_list_tmp_files], [f.name for f in rw_tmp_files]
        )

        print()

        with tempfile.NamedTemporaryFile() as sg_tmp_file:
            print_heading("Skip-Gram")
            sg_duration = skip_gram.run(
                random_walk_files=[f.name for f in rw_tmp_files],
                output_file=sg_tmp_file.name,
                dimensions=128,
                context_size=10,
                epochs=100,
                workers=n_cpus,
            )
            print()

            embeddings = data_loaders.load_embeddings(
                embeddings_file=sg_tmp_file.name, use_cuda=use_cuda
            )

            print_heading("Logistic Regression")
            logistic_regression = LogisticRegression(n_in_feats=128, n_classes=2)

            if use_cuda:
                logistic_regression.cuda()

            logistic_regression.fit = time_func(logistic_regression.fit)
            lr_duration = logistic_regression.fit(
                train_labels=labels[train_mask],
                train_mask=train_mask,
                features=embeddings,
                epochs=1,
                lr=0.01,
            )
            estimate_running_time(lr_duration)

            print_heading("Random Walks + Skip-Gram + Logistic Regression")
            estimate_running_time_with_preprocessing_steps(
                rw_duration + sg_duration,
                lr_duration
            )

            print_heading("MLP")
            mlp = MLP(n_in_feats=128, n_hidden_feats=16, n_classes=2, dropout=0.1)

            if use_cuda:
                mlp.cuda()

            mlp.fit = time_func(mlp.fit)
            mlp_duration = mlp.fit(
                train_labels=labels[train_mask],
                train_mask=train_mask,
                features=embeddings,
                epochs=1,
                lr=0.01,
            )
            estimate_running_time(mlp_duration)

            print_heading("Random Walks + Skip-Gram + MLP")
            estimate_running_time_with_preprocessing_steps(
                rw_duration + sg_duration,
                mlp_duration
            )

    finally:
        for f in tmp_files:
            f.close()


def main(args):
    n_cpus = psutil.cpu_count()
    print(f"Using all {n_cpus} available CPUs.")

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Using a single GPU.")
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}\n")
    else:
        print("GPU use is not enabled.\n")

    print("Edge list files:")
    print("\n".join(edge_lists_files))
    print()
    print("Node labels file:")
    print(train_node_labels_file)
    print()

    labels = data_loaders.load_labels(train_node_labels_file, use_cuda=args.gpu)
    n_nodes = labels.size(0)
    n_pos_nodes = (labels == 1).nonzero().size(0)
    train_mask = torch.ones(n_nodes, dtype=torch.uint8, device=labels.device)

    def estimate_running_time(
        duration,
        n_epochs=args.epochs,
        n_bootstraps=args.bootstraps,
        n_loocv_iterations=n_pos_nodes,
    ):
        duration = duration * n_epochs * n_bootstraps * n_loocv_iterations
        print_running_time(duration, n_epochs, n_bootstraps, n_loocv_iterations)

    def estimate_running_time_with_preprocessing_steps(
        preprocessing_duration,
        fit_duration,
        n_epochs=args.epochs,
        n_bootstraps=args.bootstraps,
        n_loocv_iterations=n_pos_nodes,
    ):
        duration = preprocessing_duration + (fit_duration * n_epochs * n_bootstraps * n_loocv_iterations)
        print_running_time(duration, n_epochs, n_bootstraps, n_loocv_iterations)

    with_concat_networks(
        edge_lists_files,
        labels,
        train_mask,
        n_nodes,
        n_pos_nodes,
        estimate_running_time,
        estimate_running_time_with_preprocessing_steps,
        use_cuda=args.gpu,
        n_cpus=n_cpus
    )

    with_multilayer_network(
        edge_lists_files,
        labels,
        train_mask,
        n_nodes,
        n_pos_nodes,
        estimate_running_time,
        estimate_running_time_with_preprocessing_steps,
        use_cuda=args.gpu,
        n_cpus=n_cpus
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--bootstraps", type=int, default=5, help="Number of bootstraps")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    main(args)
