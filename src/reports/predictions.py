from glob import glob
from os import path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient

from src import config


def main():
    client = MlflowClient()
    experiments = client.list_experiments()
    predict_experiment = next(ex for ex in experiments if ex.name == "Predict")
    predict_run_infos = client.list_run_infos(predict_experiment.experiment_id)

    predictions_dfs = []
    for run_info in predict_run_infos:
        run = client.get_run(run_info.run_id)

        # Build model name
        model = run.data.params["model"]

        if model == "bagging-rgcn-with-embeddings":
            edge_lists_merged_layers = run.data.params["embeddings_merged_layers"]
            model_name = (
                f"{model} | embeddings_merged_layers={edge_lists_merged_layers}"
            )
        else:
            merged_layers = run.data.params["merged_layers"]
            model_name = f"{model} | merged_layers={merged_layers}"

        node_features = run.data.params.get("node_features", "")
        if node_features:
            model_name += f", node_features={node_features}"

        # Get predictions artifact
        abs_artifact_uri = run_info.artifact_uri.replace("file://", "")
        results_artifact_uri = path.join(abs_artifact_uri, "results")
        predictions_file, *_ = glob(path.join(results_artifact_uri, "predictions*.tsv"))

        pred_df = pd.read_csv(
            predictions_file, sep="\t", index_col="node", usecols=["node", "rank"]
        )
        pred_df.rename(columns={"rank": model_name}, inplace=True)
        predictions_dfs.append(pred_df)

    prediction_df = pd.concat(predictions_dfs, axis=1)
    prediction_df.sort_values(by="node", inplace=True)

    # Save tsv file
    out_tsv_file = path.join(config.REPORTS_DIR, f"predictions.tsv")
    print(f"Saving {out_tsv_file}...")
    prediction_df.to_csv(out_tsv_file, sep="\t")

    # Create heatmap
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(
        figsize=(0.7 * prediction_df.shape[1], 0.5 * prediction_df.shape[0])
    )
    fig.suptitle("Predicted rank for each node by model.")
    sns.heatmap(prediction_df.T, annot=True, cbar=False, ax=ax)

    # Save heatmap
    out_png_file = path.join(config.REPORTS_DIR, f"predictions_heatmap.png")
    print(f"Saving {out_png_file}...")
    fig.savefig(out_png_file, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    print(
        "Generating prediction reports that will be available in the 'reports' directory."
    )
    main()
