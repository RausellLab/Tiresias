import mlflow
import torch
import ray
from src.models.propagation.rwr_m import RwrM
from src.evaluation import predict
from src.utils import data_loaders
from src.utils import mlflow as u_mlflow

RUN_NAME = "RWR-M"
MODEL_NAME = "rwr-m"


@ray.remote(num_gpus=1)
def rwr_m(
    adj_matrix_files,
    train_node_labels_file,
    test_node_labels_file,
    use_cuda,
    params,
    metadata,
):
    mlflow.set_experiment("Predict")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_param("model", MODEL_NAME)

        u_mlflow.add_params(**params)
        u_mlflow.add_metadata(metadata)
        mlflow.set_tag("use_cuda", use_cuda)

        train_labels = data_loaders.load_labels(
            train_node_labels_file, use_cuda=use_cuda
        )
        test_labels = data_loaders.load_labels(test_node_labels_file, use_cuda=use_cuda)
        labels = (train_labels.byte() | test_labels.byte()).long()

        print("Loading adjacency matrices")
        adj_matrices = data_loaders.load_adj_matrices(
            adj_matrix_files, normalization=None, use_cuda=use_cuda
        )

        adjacency_matrices = torch.stack(adj_matrices)

        print(RUN_NAME)
        ranks_df = predict.run(
            labels=labels,
            model_class=RwrM,
            adjacency_matrices=adjacency_matrices,
            **params
        )

        u_mlflow.log_dataframe(ranks_df, "predictions", "results")
