import ray
import mlflow
from src.models import DirectNeighbors
from src.evaluation import loocv
from src.utils import data_loaders
from src.utils import data_savers
from src.utils import mlflow as u_mlflow


RUN_NAME = "Direct Neighbors"
MODEL_NAME = "direct-neighbors"


@ray.remote(num_gpus=1)
def direct_neighbors(adj_matrix_file, node_labels_file, use_cuda, metadata):
    mlflow.set_experiment("LOOCV")

    with mlflow.start_run(run_name=RUN_NAME):
        u_mlflow.add_metadata(metadata)
        mlflow.set_tag("use_cuda", use_cuda)
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("merged_layers", True)

        labels = data_loaders.load_labels(node_labels_file, use_cuda=use_cuda)
        n_nodes = labels.size(0)

        graph = data_loaders.load_graph(
            [adj_matrix_file],
            n_nodes,
            self_loop=False,
            normalization=None,
            use_cuda=use_cuda,
        )

        print(RUN_NAME)
        ranks_df = loocv.run(labels=labels, model_class=DirectNeighbors, graph=graph)

        data_savers.save_ranks(ranks_df, n_nodes, RUN_NAME)
