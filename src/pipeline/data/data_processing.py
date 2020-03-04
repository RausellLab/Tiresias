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
            (io.read_edge_list(f,is_nodename_int=False) for f in config.edge_lists_files)
        )
    #Get the unique nodes, at the same time will be our reference_df

    unique_nodes_df = preprocess.unique_nodes_from_edgelist(merged_edge_lists_df)
    index_reference = unique_nodes_df['node']
    
    #TODO Reference annotation with gene symbol
    annotated_df = pd.read_csv("./data/our_protein_coding_gene_list_20760genes_collapsed.tsv", sep = "\t")
    unique_nodes_df_annotated = preprocess.reference_annotation(unique_nodes_df, annotated_df)
    
    #Save in the same directory as train_node_labels
    io.write_reference_df(unique_nodes_df_annotated, path.split(config.train_node_labels_file)[0])

    #Process node_labels
    for node_labels_path in [config.test_node_labels_file,config.train_node_labels_file]:
        node_labels_df = io.read_node_labels(node_labels_path, is_nodelist_full=False)
        label_processed_df = preprocess.label_df_generation(unique_nodes_df, node_labels_df)
        label_processed_df_reindex = preprocess.reindex_labels_df_process(label_processed_df)

        io.write_processed_df(label_processed_df_reindex, node_labels_path)
    
    #Process edge_lists
    for edge_list_path in config.edge_lists_files:
        edge_list_df = io.read_edge_list(edge_list_path, is_nodename_int=False)
        edge_list_df_reindexed = preprocess.reindex_edgelist_df_process(edge_list_df, index_reference)
        io.write_processed_df(edge_list_df_reindexed, edge_list_path)

    #Process node_attributes
    node_attributes_path = config.node_attributes_file
    node_attributes_df = io.read_node_attributes(node_attributes_path, nodes_as_index=False)
    node_attributes_processed_df = preprocess.label_process(index_reference, node_attributes_df)
    node_attributes_reindexed_df = preprocess.reindex_nodeatt_df_process(node_attributes_processed_df, index_reference)
    io.write_processed_df(node_attributes_reindexed_df, node_attributes_path)
    
if __name__ == "__main__":
    main()    