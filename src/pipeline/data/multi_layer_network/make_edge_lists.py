from os import path
from src import config
from src.config import artifact_stores
from src.utils import io


def main():
    container = artifact_stores.edge_lists.create_artifact_container()
    container.save_params(
        source_files=[path.basename(f) for f in config.edge_lists_files_processed],
        merged_layers=False,
    )

    for index, file in enumerate(config.edge_lists_files_processed):
        edge_list_df = io.read_edge_list(file)

        outfile = container.create_artifact_filepath(f"edge_list_{index}.tsv")
        # Write SNAP compatible edge list, edgelist will be 0 based
        edge_list_df.iloc[:, 0:2] += 1
        edge_list_df.to_csv(outfile, sep=" ", header=False, index=False)


if __name__ == "__main__":
    main()
