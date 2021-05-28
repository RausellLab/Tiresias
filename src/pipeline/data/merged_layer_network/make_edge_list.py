from os import path
from src import config
from src.config import artifact_stores
from src.utils.preprocess import merge_edge_lists
from src.utils import io

OUTFILE_NAME = "edge_list.tsv"


def main():
    merged_edge_lists_df = merge_edge_lists(
        (io.read_edge_list(f) for f in config.edge_lists_files_processed)
    )

    container = artifact_stores.edge_lists.create_artifact_container()
    container.save_params(
        source_files=[path.basename(f) for f in config.edge_lists_files_processed],
        merged_layers=True,
    )
    outfile = container.create_artifact_filepath(OUTFILE_NAME)
    # Write pecanpy compatible edge list
    merged_edge_lists_df.to_csv(outfile, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
