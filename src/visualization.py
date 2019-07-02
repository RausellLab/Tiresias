import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use("seaborn")


def plot_cumulative_distribution(x_nodes, y_cum_dist, auc, name, params=None):
    if params is None:
        params = {}

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    ax.set_title("Cumulative distribution curve")
    ax.set_xlabel("Ranks")
    ax.set_ylabel("Cumulative distribution")

    params = "\n".join([f"{key}: {value}" for key, value in params.items()])
    label = f"{name}\n\n" \
            f"Parameters\n" \
            f"--\n" \
            f"{params}\n\n" \
            f"Metrics\n" \
            f"--\n" \
            f"AUC: {auc:.3f}"
    ax.plot(x_nodes, y_cum_dist, label=label)
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")

    plt.tight_layout()

    return fig
