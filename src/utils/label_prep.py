import pandas as pd
import numpy as np


def read_node_labels(file):
    return pd.read_csv(
        file,
        header=None,
        names=["node", "label"],
        skiprows=1,
        sep="\t",
        dtype={"node": np.int64, "label": np.int64},
    )


def read_edge_list(file):
    return pd.read_csv(
        file,
        header=None,
        names=["src", "dst", "weight"],
        skiprows=1,
        sep="\t",
        dtype={"src": np.int64, "dst": np.int64, "weight": np.float64},
    )


def merge_edge_lists(edge_list_dfs):
    """Merge edge lists. Weights of overlapping edges are summed.

    Parameters
    ----------
    edge_list_dfs: a sequence of DataFrame
        The DataFrame containing the edge lists.
    """

    # Concatenate DataFrame
    concat_edge_list_df = pd.concat(edge_list_dfs, ignore_index=True)

    # Sum the weights of the rows that have duplicated edges
    concat_edge_list_df = concat_edge_list_df.groupby(["src", "dst"], as_index=False)[
        "weight"
    ].sum()

    return concat_edge_list_df

def unique_nodes_from_edgelist(edge_list_df):
    unique = pd.unique(pd.concat([edge_list_df['src'], edge_list_df['dst']], axis = 0))
    
    return np.sort(unique)
 


def label_process(unique_nodes, labels_df):
    
    return labels_df.loc[labels_df['node'].isin(unique_nodes)]



def write_processed_labels(labels_process_df):
    labels_process_df.to_csv("/home/agarcia/Downloads/hola.tsv".replace(".tsv", "_processed.tsv"),sep = "\t" , index = False)


def read_node_attributes(file):
    node_attrib_df = pd.read_csv(file, sep="\t", index_col=[0], dtype=float)
    node_attrib_df_type = node_attrib_df.astype({'node': np.int64})
    
    return node_attrib_df_type

    

if __name__ == "__main__":
    node_labels_df = read_node_labels("/home/agarcia/Tiresias_01/Tiresias/data/train_node_labels.tsv")
    edge_list_df = read_edge_list("/home/agarcia/Tiresias_01/Tiresias/data/edge_lists/layer_0.tsv")
    
    unique_nodes_df = unique_nodes_from_edgelist(edge_list_df)
    label_processed_df = label_process(unique_nodes_df, node_labels_df)
    

    write_processed_labels(label_processed_df)


    a = 0