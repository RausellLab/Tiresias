import ray
from functools import partial
from src import config
from src.config import artifact_stores
from src.utils.parameters import param_combinations
from src.pipeline.test.tasks.direct_neighbors import direct_neighbors
from src.pipeline.test.tasks.net_prop_with_restart import label_spreading
from src.pipeline.test.tasks.net_prop_with_restart import rwr
from src.pipeline.test.tasks.bagging_logistic_regression import (
    bagging_logistic_regression,
)
from src.pipeline.test.tasks.bagging_mlp import bagging_mlp
from src.pipeline.test.tasks.bagging_gcn import bagging_gcn
from src.pipeline.test.tasks.bagging_rgcn import bagging_rgcn
from src.pipeline.test.tasks.rwr_m import rwr_m
from src.pipeline.test.tasks.bagging_rgcn_embeddings import bagging_rgcn_embeddings


param_combinations = partial(param_combinations, "test")


def main():
    if config.gpus > 0:
        use_cuda = True
    else:
        config.gpus = 1
        use_cuda = False

    ray.init(
        num_cpus=config.cpus,
        num_gpus=config.gpus,
        resources={"memory": config.memory},
        temp_dir=config.temp_dir,
    )

    results = []
    train_node_labels_file = config.train_node_labels_file
    test_node_labels_file = config.test_node_labels_file
    node_attributes_file = config.node_attributes_file

    for (
        adjacency_matrix_files,
        metadata,
    ) in artifact_stores.adjacency_matrices.merged_layer:
        adjacency_matrix_file = adjacency_matrix_files[0]

        if config.models["direct_neighbors"]:
            dn_res = direct_neighbors.remote(
                adjacency_matrix_file,
                train_node_labels_file,
                test_node_labels_file,
                use_cuda,
                metadata,
            )
            results.append(dn_res)

        if config.models["label_spreading"]:
            for params in param_combinations("label_spreading"):
                ls_res = label_spreading.remote(
                    adjacency_matrix_file,
                    train_node_labels_file,
                    test_node_labels_file,
                    use_cuda,
                    params,
                    metadata,
                )
                results.append(ls_res)

        if config.models["rwr"]:
            for params in param_combinations("rwr"):
                rwr_res = rwr.remote(
                    adjacency_matrix_file,
                    train_node_labels_file,
                    test_node_labels_file,
                    use_cuda,
                    params,
                    metadata,
                )
                results.append(rwr_res)

        if config.models["bagging_gcn"]:
            for params in param_combinations("bagging_gcn"):
                bagging_gcn_res = bagging_gcn.remote(
                    adjacency_matrix_file,
                    None,
                    train_node_labels_file,
                    test_node_labels_file,
                    use_cuda,
                    params,
                    metadata,
                )
                results.append(bagging_gcn_res)

        if config.models["bagging_gcn_with_attributes"]:
            for params in param_combinations("bagging_gcn_with_attributes"):
                bagging_gcn_att_res = bagging_gcn.remote(
                    adjacency_matrix_file,
                    node_attributes_file,
                    train_node_labels_file,
                    test_node_labels_file,
                    use_cuda,
                    params,
                    metadata,
                )
                results.append(bagging_gcn_att_res)

    for (
        adjacency_matrix_files,
        metadata,
    ) in artifact_stores.adjacency_matrices.multi_layer:

        if config.models["rwr_m"]:
            for params in param_combinations("rwr_m"):
                rwr_m_res = rwr_m.remote(
                    adjacency_matrix_files,
                    train_node_labels_file,
                    test_node_labels_file,
                    False,
                    params,
                    metadata,
                )
                results.append(rwr_m_res)

        if config.models["bagging_rgcn"]:
            for params in param_combinations("bagging_rgcn"):
                bagging_rgcn_res = bagging_rgcn.remote(
                    adjacency_matrix_files,
                    train_node_labels_file,
                    test_node_labels_file,
                    None,
                    use_cuda,
                    params,
                    metadata,
                )
                results.append(bagging_rgcn_res)

        if config.models["bagging_rgcn_with_attributes"]:
            for params in param_combinations("bagging_rgcn_with_attributes"):
                bagging_rgcn_att_res = bagging_rgcn.remote(
                    adjacency_matrix_files,
                    train_node_labels_file,
                    test_node_labels_file,
                    node_attributes_file,
                    use_cuda,
                    params,
                    metadata,
                )
                results.append(bagging_rgcn_att_res)

    for embeddings_files, metadata in artifact_stores.embeddings:
        embeddings_file = embeddings_files[0]

        if config.models["bagging_logistic_regression"]:
            for bagging_log_reg_params in param_combinations(
                "bagging_logistic_regression"
            ):
                lg_res = bagging_logistic_regression.remote(
                    embeddings_file,
                    train_node_labels_file,
                    test_node_labels_file,
                    None,
                    use_cuda,
                    bagging_log_reg_params,
                    metadata,
                )
                results.append(lg_res)

        if config.models["bagging_logistic_regression_with_attributes"]:
            for bagging_log_reg_params in param_combinations(
                "bagging_logistic_regression_with_attributes"
            ):
                lg_att_res = bagging_logistic_regression.remote(
                    embeddings_file,
                    train_node_labels_file,
                    test_node_labels_file,
                    node_attributes_file,
                    use_cuda,
                    bagging_log_reg_params,
                    metadata,
                )
                results.append(lg_att_res)

        if config.models["bagging_mlp"]:
            for bagging_mlp_params in param_combinations("bagging_mlp"):
                mlp_res = bagging_mlp.remote(
                    embeddings_file,
                    train_node_labels_file,
                    test_node_labels_file,
                    None,
                    use_cuda,
                    bagging_mlp_params,
                    metadata,
                )
                results.append(mlp_res)

        if config.models["bagging_mlp_with_attributes"]:
            for bagging_mlp_params in param_combinations("bagging_mlp_with_attributes"):
                mlp_res = bagging_mlp.remote(
                    embeddings_file,
                    train_node_labels_file,
                    test_node_labels_file,
                    node_attributes_file,
                    use_cuda,
                    bagging_mlp_params,
                    metadata,
                )
                results.append(mlp_res)

    if config.models["bagging_rgcn_with_embeddings"]:
        for (
            adjacency_matrix_files,
            metadata_adj,
        ) in artifact_stores.adjacency_matrices.multi_layer:
            for embeddings_files, metadata_embed in artifact_stores.embeddings:
                embeddings_file = embeddings_files[0]

                metadata = {
                    "edge_lists": metadata_embed,
                    "adjacency_matrices": metadata_adj,
                }

                for params in param_combinations("bagging_rgcn_with_embeddings"):
                    res = bagging_rgcn_embeddings.remote(
                        adjacency_matrix_files,
                        train_node_labels_file,
                        test_node_labels_file,
                        embeddings_file,
                        use_cuda,
                        params,
                        metadata,
                    )
                    results.append(res)

    ray.get(results)


if __name__ == "__main__":
    main()
