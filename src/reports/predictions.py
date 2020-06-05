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

        #If some models failed, not take into account empty folders
        try:
            predictions_file, *_ = glob(path.join(results_artifact_uri, "predictions*.tsv"))
        except:
            next

        pred_df = pd.read_csv(
            predictions_file, sep="\t", index_col="node", usecols=["node", "rank"]
        )
        pred_df.rename(columns={"rank": model_name}, inplace=True)
        predictions_dfs.append(pred_df)

    prediction_df = pd.concat(predictions_dfs, axis=1)
    prediction_df.sort_values(by="node", inplace=True)
    
    #Remove duplicated columns
    prediction_df = prediction_df.loc[:,~prediction_df.columns.duplicated()]

    #Rename nodes taking the indexes reference
    index_reference_path = path.join(path.split(config.train_node_labels_file)[0],"reference_index.tsv")
    index_reference = pd.read_csv(index_reference_path, sep = "\t", index_col=[0])
    new_index = index_reference["node"].values[prediction_df.index.values]
    #Taking the list of gene_symbols
    gene_symbol = index_reference["Gene"].values[prediction_df.index.values]

    prediction_df_reindex = prediction_df.set_index(new_index)
    prediction_df_reindex.index.name = "node"
    try:

        # Create heatmap
        sns.set(font_scale=1.5)
        fig, ax = plt.subplots(
            figsize=(0.7 * prediction_df_reindex.shape[1], 0.5 * prediction_df_reindex.shape[0])
        )
        fig.suptitle("Predicted rank for each node by model.")
        sns.heatmap(prediction_df_reindex.T, annot=True, cbar=False, ax=ax)

        # Save heatmap
        out_png_file = path.join(config.REPORTS_DIR, f"predictions_heatmap.png")
        print(f"Saving {out_png_file}...")
        fig.savefig(out_png_file, bbox_inches="tight", dpi=150)
    except:
        print('Image size exceed dimensions, too many datapoints. Continuing predictions report generation ')

    #Annotating predictions with gene_symbols

    prediction_df_reindex.insert(loc=0, column='Gene_symbols', value=gene_symbol)
    # Save tsv file
    out_tsv_file = path.join(config.REPORTS_DIR, f"predictions.tsv")
    print(f"Saving {out_tsv_file}...")
    prediction_df_reindex.to_csv(out_tsv_file, sep="\t")

if __name__ == "__main__":
    print(
        "Generating prediction reports that will be available in the 'reports' directory."
    )
    main()
