import ray
import mlflow
import torch
from src.models import Bagging
from src.models import LogisticRegression
from src.evaluation import loocv
from src.utils import data_loaders
from src.utils import data_savers
from src.utils import mlflow as u_mlflow


RUN_NAME = "Bagging Logistic Regression"
MODEL_NAME = "bagging-logistic-regression"


#@ray.remote(num_gpus=1)
def bagging_logistic_regression(
    embeddings_file, node_labels_file, node_features_file, use_cuda, params, metadata
):
    mlflow.set_experiment("LOOCV")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_param("model", MODEL_NAME)

        u_mlflow.add_params(**params)
        u_mlflow.add_metadata(metadata)
        mlflow.set_tag("use_cuda", use_cuda)

        labels = data_loaders.load_labels(node_labels_file, use_cuda=use_cuda)
        n_nodes = labels.size(0)

        embeddings = None
        node_features = None

        if embeddings_file is not None:
            mlflow.log_param("embeddings", True)
            mlflow.log_artifact(embeddings_file, "inputs")
            embeddings = data_loaders.load_embeddings(
                embeddings_file, use_cuda=use_cuda
            )
        else:
            mlflow.log_param("embeddings", False)

        if node_features_file is not None:
            mlflow.log_param("node_features", True)
            mlflow.log_artifact(node_features_file, "inputs")
            node_features = data_loaders.load_node_features(
                node_features_file, use_cuda
            )
        else:
            mlflow.log_param("node_features", False)

        if embeddings is not None and node_features is not None:
            in_features = torch.cat((embeddings, node_features), dim=1)
        elif embeddings is not None:
            in_features = embeddings
        elif node_features is not None:
            in_features = node_features

        print(RUN_NAME)
        ranks_df = loocv.run(
            labels=labels,
            model_class=Bagging,
            bagging_model=LogisticRegression,
            features=in_features,
            **params
        )

        data_savers.save_ranks(ranks_df, n_nodes, RUN_NAME, params)
