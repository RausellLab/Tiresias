import pandas as pd
import numpy as np
from os import path
from src import config
from src.utils import preprocess
from src.utils import io
from src import config

def main():
    
    merged_edge_lists_df = preprocess.merge_edge_lists(
            (io.read_edge_list(f) for f in config.edge_lists_files)
        )
    for node_labels_path in [config.train_node_labels_file, config.test_node_labels_file ]:
        
        node_labels_df = io.read_node_labels(node_labels_path)
        unique_nodes_df = preprocess.unique_nodes_from_edgelist(merged_edge_lists_df)
        label_processed_df = preprocess.label_process(unique_nodes_df, node_labels_df)

        io.write_processed_labels(label_processed_df, node_labels_path)



if __name__ == "__main__":
    main()    