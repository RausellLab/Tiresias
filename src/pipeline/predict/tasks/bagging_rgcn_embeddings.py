import mlflow
import ray
from src.models import Bagging
from src.models import RGCN
from src.evaluation import predict
from src.utils import data_loaders
from src.utils import mlflow as u_mlflow


RUN_NAME = "Bagging RGCN with embeddings"
MODEL_NAME = "bagging-rgcn-with-embeddings"
NORMALIZATION = "sym"
SELF_LOOP = True


#@ray.remote(num_gpus=1)
def bagging_rgcn_embeddings(
    adj_matrix_files,
    train_node_labels_file,
    test_node_labels_file,
    embeddings_file,
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
        n_nodes = labels.size(0)

        embeddings = data_loaders.load_embeddings(embeddings_file, use_cuda=use_cuda)

        graph = data_loaders.load_graph(
            adj_matrix_files,
            n_nodes,
            add_edge_type=True,
            add_node_ids=True,
            normalization=NORMALIZATION,
            use_cuda=use_cuda,
        )

        print(RUN_NAME)
        ranks_df = predict.run(
            labels=labels,
            model_class=Bagging,
            bagging_model=RGCN,
            features=embeddings,
            graph=graph,
            n_rels=len(adj_matrix_files),
            **params
        )

        u_mlflow.log_dataframe(ranks_df, "predictions", "results")
