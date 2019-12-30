import json
from functools import reduce
from glob import glob
from os import path

# from textwrap import wrap
import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from scipy import stats

from src import config
from src.utils.misc import product_dict

models = [
    {
        "name": "bagging-rgcn-with-embeddings",
        "params": {"embeddings_merged_layers": [True, False]},
    },
    {
        "name": "bagging-mlp",
        "params": {"merged_layers": [True, False], "node_features": [True, False]},
    },
    {
        "name": "bagging-logistic-regression",
        "params": {"merged_layers": [True, False], "node_features": [True, False]},
    },
    {
        "name": "bagging-rgcn",
        "params": {"merged_layers": [True, False], "node_features": [True, False]},
    },
    {"name": "rwr-m", "params": {"merged_layers": [True, False],}},
    {
        "name": "bagging-gcn",
        "params": {"merged_layers": [True], "node_features": [True, False]},
    },
    {"name": "rwr", "params": {"merged_layers": [True],}},
    {"name": "label-spreading", "params": {"merged_layers": [True],}},
    {"name": "direct-neighbors", "params": {"merged_layers": [True],}},
]


def get_best_runs_by_model(experience_name):
    client = MlflowClient()
    experiments = client.list_experiments()
    experiment = next(ex for ex in experiments if ex.name == experience_name)
    experiment_id = experiment.experiment_id

    best_runs_by_model = []
    for model in models:
        for params in product_dict(**model["params"]):
            filter_string = f'params.model="{model["name"]}"'

            for param_key, param_value in params.items():
                filter_string += f' and params.{param_key}="{param_value}"'

            runs = client.search_runs(
                experiment_id,
                filter_string,
                order_by=["metric.auc DESC"],
                max_results=1,
            )

            if not runs:
                continue

            run = runs[0]

            run_name = run.data.tags["mlflow.runName"]
            for idx, (param_key, param_value) in enumerate(params.items()):
                if idx == 0:
                    run_name += f" | "
                else:
                    run_name += f", "
                run_name += f"{param_key}={param_value}"

            best_runs_by_model.append({"name": run_name, "run": runs[0]})

    return best_runs_by_model


def dict_to_html(dictionary):
    return "<br>".join([f"{key}={value}" for key, value in dictionary.items()])


def save_runs(runs, experience_name):
    data = {
        "Run ID": [],
        "Run Name": [],
        "Model": [],
        "Parameters": [],
        "Metrics": [],
        "AUC": [],
        "AUC Ratio": [],
    }

    for item in runs:
        run = item["run"]
        run_name = item["name"]

        data["Run ID"].append(run.info.run_id)
        data["Run Name"].append(run_name)
        data["Model"].append(run.data.params["model"])
        data["Parameters"].append(run.data.params)
        data["Metrics"].append(run.data.metrics)
        data["AUC"].append(run.data.metrics["auc"])
        data["AUC Ratio"].append(run.data.metrics["auc_ratio"])

    runs_df = pd.DataFrame(
        data,
        columns=[
            "Run ID",
            "Run Name",
            "Model",
            "Parameters",
            "Metrics",
            "AUC",
            "AUC Ratio",
        ],
        dtype=object,
    )
    runs_df.sort_values(by="AUC", ascending=False, inplace=True)

    runs_df_tsv = runs_df.copy()
    runs_df_tsv["Parameters"] = runs_df_tsv["Parameters"].apply(json.dumps)
    runs_df_tsv["Metrics"] = runs_df_tsv["Metrics"].apply(json.dumps)
    out_tsv_file = path.join(
        config.REPORTS_DIR, f"{experience_name.lower()}_best-runs-by-model.tsv"
    )
    print(f"Saving {out_tsv_file}...")
    runs_df_tsv.to_csv(out_tsv_file, sep="\t")

    runs_df_html = runs_df.copy()
    runs_df_html["Parameters"] = runs_df_html["Parameters"].apply(dict_to_html)
    runs_df_html["Metrics"] = runs_df_html["Metrics"].apply(dict_to_html)
    out_html_file = path.join(
        config.REPORTS_DIR, f"{experience_name.lower()}_best-runs-by-model.html"
    )
    print(f"Saving {out_html_file}...")
    pd.set_option("display.max_colwidth", -1)
    runs_df_html.to_html(out_html_file, justify="justify", escape=False)


def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    rho, pval = stats.spearmanr(x, y)
    ax = ax or plt.gca()
    ax.annotate(
        f"$\\rho$ = {rho:.2f}, pval={pval:.2f}", xy=(0.1, 0.9), xycoords=ax.transAxes
    )


def generate_pair_grid(runs, experience_name):
    ranks_dfs = []
    for item in runs:
        run = item["run"]
        run_name = item["name"]

        # Get ranks artifact
        abs_artifact_uri = run.info.artifact_uri.replace("file://", "")
        results_artifact_uri = path.join(abs_artifact_uri, "results")
        ranks_file, *_ = glob(path.join(results_artifact_uri, "ranks*.tsv"))

        # Load file with ranks
        df = pd.read_csv(ranks_file, sep="\t")
        run_name = run_name.replace(" | ", "\n")
        run_name = run_name.replace(", ", "\n")
        # run_name = "\n".join(wrap(run_name, 25))
        df.rename(columns={"rank": run_name}, inplace=True)

        ranks_dfs.append(df)

    ranks_df = reduce(
        lambda left, right: pd.merge(left, right, on="pos_node_index"), ranks_dfs
    )
    ranks_df = ranks_df.set_index("pos_node_index")

    plt.style.use("seaborn")
    plot = sns.pairplot(ranks_df, kind="reg", height=5)

    max_rank = ranks_df.max().max()
    plot.set(ylim=(0, max_rank + 1))
    plot.set(xlim=(0, max_rank + 1))
    plot.map_offdiag(corrfunc)

    # Save pair plot
    out_png_file = path.join(
        config.REPORTS_DIR, f"{experience_name.lower()}_best-runs-by-model_pairplot.png"
    )
    print(f"Saving {out_png_file}...")
    plot.savefig(out_png_file)


def main(experience_name):
    runs = get_best_runs_by_model(experience_name)
    save_runs(runs, experience_name)
    generate_pair_grid(runs, experience_name)


@click.group()
def cli():
    pass


@click.command()
def validation():
    main("LOOCV")


@click.command()
def test():
    main("Test")


cli.add_command(validation)
cli.add_command(test)


if __name__ == "__main__":
    cli()
