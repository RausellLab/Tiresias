import pandas as pd


def merge_edge_lists(edge_list_dfs):
    """
    Merge edge lists. Weights of overlapping edges are summed.

    Parameters
    ----------
    edge_list_dfs: a sequence of DataFrame
        The DataFrame containing the edge lists.
    """

    # Concatenate DataFrame
    concat_edge_list_df = pd.concat(edge_list_dfs, ignore_index=True)

    # Sum the weights of the rows that have duplicated edges
    concat_edge_list_df = concat_edge_list_df.groupby(["src", "dst"], as_index=False)["weight"].sum()

    return concat_edge_list_df
