# coding: utf-8

import mlflow
import tempfile
import numpy as np
import pandas as pd


def add_artifacts(*args):
    """
    Log multiple local files as artifacts of the currently active run.

    Parameters
    ----------
    args:
         Path to the files to write.
    """
    for local_path in args:
        mlflow.log_artifact(local_path, "inputs")


def add_params(**kwargs):
    """
    Log multiple parameters under the current run.

    Parameters
    ----------
    kwargs:
        Parameters to log.
    """
    for key, value in kwargs.items():
        mlflow.log_param(key, value)


def log_dataframe(dataframe, filename, directory=None):
    with tempfile.NamedTemporaryFile(prefix=f"{filename}_", suffix=".tsv") as tmp_file:
        dataframe.to_csv(tmp_file.name, sep="\t")
        mlflow.log_artifact(tmp_file.name, directory)


def log_ndarray(ndarray, filename, directory=None):
    with tempfile.NamedTemporaryFile(prefix=f"{filename}_", suffix=".txt") as tmp_file:
        np.savetxt(tmp_file.name, ndarray, delimiter="\t")
        mlflow.log_artifact(tmp_file.name, directory)


def log_fig(fig, filename, directory=None):
    with tempfile.NamedTemporaryFile(prefix=f"{filename}_", suffix=".png") as tmp_file:
        fig.savefig(tmp_file.name, format="png")
        mlflow.log_artifact(tmp_file.name, directory)


def flatten_dict(data):
    df = pd.io.json.json_normalize(data, sep='_')
    return df.to_dict(orient="records")[0]


def add_metadata(metadata):
    add_params(
        **flatten_dict(metadata)
    )
