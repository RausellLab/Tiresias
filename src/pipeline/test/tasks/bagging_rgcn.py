import mlflow
import ray
from src.models import Bagging
from src.models import RGCN
from src.evaluation import test
from src.utils import data_loaders
from src.utils import data_savers
from src.utils import mlflow as u_mlflow


RUN_NAME = "Bagging RGCN"
MODEL_NAME = "bagging-rgcn"
NORMALIZATION = "sym"
SELF_LOOP = True


@ray.remote(num_gpus=1)
def bagging_rgcn(
    adj_matrix_files,
    train_node_labels_file,
    test_node_labels_file,
    node_features_file,
    use_cuda,
    params,
    metadata,
):
    mlflow.set_experiment("Test")

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
        train_mask = ~test_labels.byte()
        n_nodes = labels.size(0)

        if node_features_file is not None:
            mlflow.log_param("node_features", True)
            mlflow.log_artifact(node_features_file, "inputs")
            features = data_loaders.load_node_features(node_features_file, use_cuda)
        else:
            mlflow.log_param("node_features", False)
            features = None

        graph = data_loaders.load_graph(
            adj_matrix_files,
            n_nodes,
            add_edge_type=True,
            add_node_ids=True,
            normalization=NORMALIZATION,
            use_cuda=use_cuda,
        )

        print(RUN_NAME)
        ranks_df = test.run(
            labels=labels,
            train_mask=train_mask,
            model_class=Bagging,
            bagging_model=RGCN,
            features=features,
            graph=graph,
            n_rels=len(adj_matrix_files),
            **params
        )

        data_savers.save_ranks(ranks_df, n_nodes, RUN_NAME, params)
