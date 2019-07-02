# coding: utf-8

import mlflow
import numpy as np
from src.utils.metrics import cumulative_distribution_function
from src.visualization import plot_cumulative_distribution
from src.utils import mlflow as u_mlflow
from sklearn import metrics
from matplotlib import pyplot as plt


def save_ranks(ranks_df, n_nodes, run_name, params=None):
    u_mlflow.log_dataframe(ranks_df, "ranks", "results")

    cum_dist = cumulative_distribution_function(
        ranks=ranks_df["rank"].values,
        n_nodes=n_nodes
    )

    u_mlflow.log_ndarray(cum_dist, "cum_dist", "results")

    for index, n in enumerate(cum_dist):
        mlflow.log_metric("cum_dist", n, step=index)

    # AUC
    x = np.arange(1, n_nodes + 1)
    auc = metrics.auc(x=x, y=cum_dist)
    mlflow.log_metric("auc", auc)

    # AUC ratio
    mlflow.log_metric("auc_ratio", auc / (n_nodes - 1))

    # Median rank
    median = np.median(ranks_df["rank"].values)
    mlflow.log_metric("median_rank", median)

    # Plot
    fig = plot_cumulative_distribution(x, cum_dist, auc, run_name, params)
    u_mlflow.log_fig(fig, "cumulative_distribution", "figures")
    plt.close(fig)
