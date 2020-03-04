import pandas as pd
import numpy as np

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
    unique_sorted = np.sort(unique)
    unique_df = pd.DataFrame(data=unique_sorted, columns = ['node'])
    return unique_df

def reference_annotation(reference_df, annotated_df):
    """Return the annotated reference dataframe, given a previous annotated df with a column called gene(which will contain the ensemblID)
    Parameters
    ----------
    reference_df: a DataFrame
        The DataFrame containing the sorted and unique edge list, wich index is will be the name of the nodes, column called node
    annotated_df: a DataFrame
        The DataFrame containing a column called Gene_ID that contains the names of nodes and the respective annotation in another columns.
    """
    reference_annotated_df = pd.merge(reference_df, annotated_df, left_on = 'node', right_on = 'Gene_ID', how = 'left')
    return reference_annotated_df.drop(['Gene_ID'], axis = 1)

def label_process(unique_nodes, labels_df):
    """returns a datframe with the nodes in labels lists that are inside our graphs

    Parameters
    ----------
    unique_nodes: a DataFrame
        The DataFrame containing the sorted and unique edge list.
    labels_df: a DataFrame
        The DataFrame containing node and labels.
    """
    return labels_df.loc[labels_df['node'].isin(unique_nodes)]

def label_df_generation(reference_df, label_df):
    """Return a dataframe with all of values present in node columns in reference and 
    Parameters
    ----------
    reference_df: a DataFrame
        The DataFrame containing the sorted and unique edge list, wich index is will be the name of the nodes.
    labels_df: a DataFrame
        The DataFrame containing only node names.
    """
    label_df_processed = reference_df
    label_df_processed["label"] = 0
    label_df_processed.loc[reference_df["node"].isin(label_df['node']),'label'] = 1
    return label_df_processed

def reindex_labels_df_process(node_labels_df):
    """
    Takes node_labels pd.Dataframe and rename the nodes by his indexes.
    """
    node_labels_df_processed = node_labels_df.copy()
    node_labels_df_processed['node'] = node_labels_df_processed.index.to_numpy()
    return node_labels_df_processed

def reindex_edgelist_df_process(edge_list_df, index_reference):
    """
    Takes a labels_dataframe of labels and a the reference of the index and returns the labels_dataframe with 
    the reindexing
    """
    for columname in ['src','dst']:
        edge_list_df[columname] = column_reindexing(edge_list_df[columname], index_reference)
            
    return edge_list_df

def reindex_nodeatt_df_process(nodeatt_df, index_reference):
    """
    Takes a labels_dataframe of labels and a the reference of the index and returns the labels_dataframe with 
    the reindexing
    """
    
    nodeatt_df['node'] = column_reindexing(nodeatt_df['node'], index_reference)
    return nodeatt_df

def column_reindexing(node_column, reference_column):
    """ 
    Takes pd series of columns and a pd series of reference indexing(nodes sorted (index inside))
    Change the id of the values in pandas_columns to the correspondent index in the reference dataframe. 
    Return the pd.Series of the with the reindexed(renamed nodes).
    Parameters
    ----------
    node_column: Pandas series 
        The Column containing the nodes to rename.
    reference_column: 
        The column of indexed nodes. Reference for the change.
    """ 
    column = node_column.values.tolist()
    reference = reference_column.tolist()
    return pd.Series(data = [reference.index(x) for x in column], dtype=np.int64, name = node_column.name)

