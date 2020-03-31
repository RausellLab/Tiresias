import ray
import mlflow
from src.models import NetPropWithRestart
from src.evaluation import loocv
from src.utils import data_loaders
from src.utils import data_savers
from src.utils import mlflow as u_mlflow


def net_prop_with_restart(
    run_name,
    model_name,
    normalization,
    adj_matrix_file,
    node_labels_file,
    use_cuda,
    params,
    metadata,
):
    mlflow.set_experiment("LOOCV")

    with mlflow.start_run(run_name=run_name):
        u_mlflow.add_params(**params)
        u_mlflow.add_metadata(metadata)

        mlflow.log_param("model", model_name)
        mlflow.set_tag("use_cuda", use_cuda)
        mlflow.log_param("merged_layers", True)

        labels = data_loaders.load_labels(node_labels_file, use_cuda=use_cuda)
        n_nodes = labels.size(0)

        adjacency_matrix = data_loaders.load_adj_matrices(
            [adj_matrix_file], normalization=normalization, use_cuda=use_cuda
        )[0]

        print(run_name)
        ranks_df = loocv.run(
            labels=labels,
            model_class=NetPropWithRestart,
            adjacency_matrix=adjacency_matrix,
            **params
        )

        data_savers.save_ranks(ranks_df, n_nodes, run_name, params)


#@ray.remote(num_gpus=1)
def rwr(adj_matrix_file, node_labels_file, use_cuda, params, metadata):
    net_prop_with_restart(
        "Random walk with restart",
        "rwr",
        "rw",
        adj_matrix_file,
        node_labels_file,
        use_cuda,
        params,
        metadata,
    )


#@ray.remote(num_gpus=1)
def label_spreading(adj_matrix_file, node_labels_file, use_cuda, params, metadata):
    net_prop_with_restart(
        "Label Spreading",
        "label-spreading",
        "sym",
        adj_matrix_file,
        node_labels_file,
        use_cuda,
        params,
        metadata,
    )
