import pandas as pd
import numpy as np
from os import path
from src import config
from src.utils import preprocess
from src.utils import io
from src import config

def main():
    #Merge the lists
    merged_edge_lists_df = preprocess.merge_edge_lists(
            (io.read_edge_list(f) for f in config.edge_lists_files)
        )
    #Get the unique nodes, at the same time will be our reference_df

    unique_nodes_df = preprocess.unique_nodes_from_edgelist(merged_edge_lists_df)
    index_reference = unique_nodes_df['node']
    
    #Save in the same directory as train_node_labels
    io.write_reference_df(unique_nodes_df, path.split(config.train_node_labels_file)[0])

    #Process node_labels
    for node_labels_path in [config.train_node_labels_file, config.test_node_labels_file]:
        node_labels_df = io.read_node_labels(node_labels_path)
        label_processed_df = preprocess.label_process(index_reference, node_labels_df)
        label_processed_df_reindex = preprocess.reindex_labels_df_process(label_processed_df)

        io.write_processed_df(label_processed_df_reindex, node_labels_path)
    
    #Process edge_lists
    for edge_list_path in config.edge_lists_files:
        edge_list_df = io.read_edge_list(edge_list_path)
        edge_list_df_reindexed = preprocess.reindex_edgelist_df_process(edge_list_df, index_reference)
        io.write_processed_df(edge_list_df_reindexed, edge_list_path)

    #Process node_attributes
    node_attributes_path = config.node_attributes_file
    node_attributes_df = io.read_node_attributes(node_attributes_path, nodes_as_index=False)
    node_attributes_processed_df = preprocess.label_process(index_reference, node_attributes_df)
    io.write_processed_df(node_attributes_processed_df, node_attributes_path)
    
if __name__ == "__main__":
    main()    