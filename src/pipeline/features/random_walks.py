# coding: utf-8

from src.config import artifact_stores
from src.features import node2vec_snap
from src.utils.parameters import param_combinations


def random_walks(edge_list_files, walk_length, n_walks, p, q, params):
    container = artifact_stores.random_walks.create_artifact_container()
    container.save_params(
        random_walks={
            "walk_length": walk_length,
            "n_walks": n_walks,
            "p": p,
            "q": q
        },
        **params
    )

    for index, edge_list_file in enumerate(edge_list_files):
        outfile = container.create_artifact_filepath(f"random_walks_{index}.txt")

        node2vec_snap.run(
            edge_list_file,
            outfile,
            walk_length=walk_length,
            n_walks=n_walks,
            p=p,
            q=q,
            verbose=True,
            directed_graph=True,
            weighted_graph=True,
            output_random_walks=True,
        )


def main():
    for edge_list_files, file_params in artifact_stores.edge_lists:
        for rw_params in param_combinations(stage="features", model_name="random_walks"):
            random_walks(
                edge_list_files,
                walk_length=rw_params["walk_length"],
                n_walks=rw_params["n_walks"],
                p=rw_params["p"],
                q=rw_params["q"],
                params=file_params
            )


if __name__ == "__main__":
    main()
