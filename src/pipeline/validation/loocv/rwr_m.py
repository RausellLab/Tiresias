# coding: utf-8

import mlflow
import torch
import ray
from src.models.propagation.rwr_m import RwrM
from src.evaluation import loocv
from src.utils import data_loaders
from src.utils import data_savers
from src.utils import mlflow as u_mlflow

RUN_NAME = "RWR-M"


@ray.remote(num_gpus=1)
def rwr_m(adj_matrix_files, node_labels_file, use_cuda, params, metadata):
    mlflow.set_experiment("LOOCV")

    with mlflow.start_run(run_name=RUN_NAME):
        u_mlflow.add_params(**params)
        u_mlflow.add_metadata(metadata)
        mlflow.set_tag("use_cuda", use_cuda)

        labels = data_loaders.load_labels(node_labels_file, use_cuda=use_cuda)
        n_nodes = labels.size(0)

        print("Loading adjacency matrices")
        adj_matrices = data_loaders.load_adj_matrices(
            adj_matrix_files,
            normalization=None,
            use_cuda=use_cuda
        )

        adjacency_matrices = torch.stack(adj_matrices)

        print(RUN_NAME)
        ranks_df = loocv.run(
            labels=labels,
            model_class=RwrM,
            adjacency_matrices=adjacency_matrices,
            **params
        )

        data_savers.save_ranks(ranks_df, n_nodes, RUN_NAME, params)
