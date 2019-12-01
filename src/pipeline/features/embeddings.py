from src import config
from src.config import artifact_stores
from src.features import skip_gram
from src.utils.parameters import param_combinations


def embeddings(random_walk_files, dimensions, context_size, epochs, params):
    container = artifact_stores.embeddings.create_artifact_container()
    container.save_params(
        skip_gram={
            "dimensions": dimensions,
            "context_size": context_size,
            "epochs": epochs,
        },
        **params,
    )

    outfile = container.create_artifact_filepath(f"embeddings.txt")

    skip_gram.run(
        random_walk_files=random_walk_files,
        output_file=outfile,
        dimensions=dimensions,
        context_size=context_size,
        epochs=epochs,
        workers=config.cpus,
    )


def main():
    for random_walk_files, file_params in artifact_stores.random_walks:
        for skip_gram_params in param_combinations(
            stage="features", model_name="skip_gram"
        ):
            embeddings(
                random_walk_files,
                dimensions=skip_gram_params["dimensions"],
                context_size=skip_gram_params["context_size"],
                epochs=skip_gram_params["epochs"],
                params=file_params,
            )


if __name__ == "__main__":
    main()
