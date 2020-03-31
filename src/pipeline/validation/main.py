import ray
from functools import partial
import gc
import torch
from src import config
from src.config import artifact_stores
from src.utils.parameters import param_combinations
from src.pipeline.validation.loocv.direct_neighbors import direct_neighbors
from src.pipeline.validation.loocv.net_prop_with_restart import label_spreading
from src.pipeline.validation.loocv.net_prop_with_restart import rwr
from src.pipeline.validation.loocv.bagging_logistic_regression import (
bagging_logistic_regression,
)
from src.pipeline.validation.loocv.bagging_mlp import bagging_mlp
from src.pipeline.validation.loocv.bagging_gcn import bagging_gcn
from src.pipeline.validation.loocv.bagging_rgcn import bagging_rgcn
from src.pipeline.validation.loocv.rwr_m import rwr_m
from src.pipeline.validation.loocv.bagging_rgcn_embeddings import (bagging_rgcn_embeddings,)

param_combinations = partial(param_combinations, "validation")


def main():
    if config.gpus > 0:
        use_cuda = True
    else:
        config.gpus = 1
        use_cuda = False

    # ray.init(
    #     num_cpus=config.cpus,
    #     num_gpus=config.gpus,
    #     memory=config.memory * 1e9,
    #     temp_dir=config.temp_dir,
    #     local_mode=True
    # )
    
    results = []
    node_labels_file = config.train_node_labels_file_processed
    node_attributes_file = config.node_attributes_file_processed

    for (
        adjacency_matrix_files,
        metadata,
    ) in artifact_stores.adjacency_matrices.merged_layer:
        adjacency_matrix_file = adjacency_matrix_files[0]

        if config.models["direct_neighbors"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    dn_res = direct_neighbors(
                        adjacency_matrix_file, node_labels_file, use_cuda, metadata
                        )
                    results.append(dn_res)

                    break
                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model Failed after 3 attemps, reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()

        if config.models["label_spreading"]:
            for params in param_combinations("label_spreading"):
                ls_res = label_spreading(
                    adjacency_matrix_file, node_labels_file, use_cuda, params, metadata
                )
                results.append(ls_res)

        if config.models["rwr"]:
            for params in param_combinations("rwr"):
                rwr_res = rwr(
                    adjacency_matrix_file, node_labels_file, use_cuda, params, metadata
                )
                results.append(rwr_res)

        if config.models["bagging_gcn"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    for params in param_combinations("bagging_gcn"):
                        bagging_gcn_res = bagging_gcn(
                        adjacency_matrix_file,
                        None,
                        node_labels_file,
                        use_cuda,
                        params,
                        metadata,
                        )
                        results.append(bagging_gcn_res)
                    break

                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model run failed after 3 attemps, reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()
            

        if config.models["bagging_gcn_with_attributes"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    for params in param_combinations("bagging_gcn_with_attributes"):
                        bagging_gcn_att_res = bagging_gcn(
                        adjacency_matrix_file,
                        node_attributes_file,
                        node_labels_file,
                        use_cuda,
                        params,
                        metadata,
                        )
                        results.append(bagging_gcn_att_res)
                    break

                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model Failed after 3 attemps, reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()
            

    for (
        adjacency_matrix_files,
        metadata,
    ) in artifact_stores.adjacency_matrices.multi_layer:

        if config.models["rwr_m"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    for params in param_combinations("rwr_m"):
                        rwr_m_res = rwr_m(
                        adjacency_matrix_files, node_labels_file, False, params, metadata
                    )
                    results.append(rwr_m_res)
                    break

                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model Failed after 3 attemps, reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()
            

        if config.models["bagging_rgcn"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    for params in param_combinations("bagging_rgcn"):
                        bagging_rgcn_res = bagging_rgcn(
                        adjacency_matrix_files,
                        node_labels_file,
                        None,
                        use_cuda,
                        params,
                        metadata,
                    )
                    results.append(bagging_rgcn_res)
                    break

                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model Failed after 3 attemps, reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()
            

        if config.models["bagging_rgcn_with_attributes"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    for params in param_combinations("bagging_rgcn_with_attributes"):
                        bagging_rgcn_att_res = bagging_rgcn(
                        adjacency_matrix_files,
                        node_labels_file,
                        node_attributes_file,
                        use_cuda,
                        params,
                        metadata,
                        )
                        results.append(bagging_rgcn_att_res)
                    break

                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model Failed after 3 attemps, reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()
            

    for embeddings_files, metadata in artifact_stores.embeddings:
        embeddings_file = embeddings_files[0]

        if config.models["bagging_logistic_regression"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    for bagging_log_reg_params in param_combinations("bagging_logistic_regression"):
                        lg_res = bagging_logistic_regression(
                        embeddings_file,
                        node_labels_file,
                        None,
                        use_cuda,
                        bagging_log_reg_params,
                        metadata,
                        )
                        results.append(lg_res)
                    break

                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model Failed after 3 attemps, reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()
            

        if config.models["bagging_logistic_regression_with_attributes"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    for bagging_log_reg_params in param_combinations("bagging_logistic_regression_with_attributes"):
                        lg_att_res = bagging_logistic_regression(
                        embeddings_file,
                        node_labels_file,
                        node_attributes_file,
                        use_cuda,
                        bagging_log_reg_params,
                        metadata,
                        )
                        results.append(lg_att_res)
                    break

                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model Failed after 3 attemps, reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()
            

        if config.models["bagging_mlp"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    for bagging_mlp_params in param_combinations("bagging_mlp"):
                        mlp_res = bagging_mlp(
                        embeddings_file,
                        node_labels_file,
                        None,
                        use_cuda,
                        bagging_mlp_params,
                        metadata,
                        )
                        results.append(mlp_res)
                    break

                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model Failed after 3 attemps, reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()
            

        if config.models["bagging_mlp_with_attributes"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    for bagging_mlp_params in param_combinations("bagging_mlp_with_attributes"):
                        mlp_res = bagging_mlp(
                        embeddings_file,
                        node_labels_file,
                        node_attributes_file,
                        use_cuda,
                        bagging_mlp_params,
                        metadata,
                        )
                        results.append(mlp_res)
                    break

                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model Failed after 3 attemps, try to free up memory or reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()
            

    if config.models["bagging_rgcn_with_embeddings"]:
            retry_counter = 0
            while retry_counter < 3:
                try:
                    for (adjacency_matrix_files,metadata_adj,) in artifact_stores.adjacency_matrices.multi_layer:
                        for embeddings_files, metadata_embed in artifact_stores.embeddings:
                            embeddings_file = embeddings_files[0]

                            metadata = {
                                "embeddings": metadata_embed,
                                "adjacency_matrices": metadata_adj,
                            }

                            for params in param_combinations("bagging_rgcn_with_embeddings"):
                                res = bagging_rgcn_embeddings(
                                    adjacency_matrix_files,
                                    node_labels_file,
                                    embeddings_file,
                                    use_cuda,
                                    params,
                                    metadata,
                                )
                                results.append(res)
                    break

                except RuntimeError as e:
                    retry_counter += 1
                    print('Runtime Error {}\nRun Again......{}/{}'.format(e,retry_counter, 3))
                    if retry_counter == 3:
                        print('Model Failed after 3 attemps, reduce the amount of input data. Following pipeline')
                        break
                finally:
                    # Handle CUDA OOM Error Safely
                    gc.collect()
                    torch.cuda.empty_cache()
        

    #ray.get(results)


if __name__ == "__main__":
    main()
