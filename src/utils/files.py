import os
import json
from uuid import uuid4
from glob import glob

CONFIG_FILENAME = "params.json"


def gen_artifact_dir(parent_dir):
    dir_path = os.path.abspath(os.path.join(parent_dir, uuid4().hex))
    os.mkdir(dir_path)
    return dir_path


def save_params(directory, data):
    outfile_path = os.path.join(directory, CONFIG_FILENAME)
    with open(outfile_path, "w") as outfile:
        json.dump(data, outfile, sort_keys=True, indent=2)


def load_params(directory):
    infile_path = os.path.join(directory, CONFIG_FILENAME)
    with open(infile_path, "r") as infile_path:
        return json.load(infile_path)


def get_subdirectories(directory):
    subdirectories = next(os.walk(directory))[1]
    return [os.path.join(directory, subdirectory) for subdirectory in subdirectories]


def iter_artifact_dir(directory):
    for subdirectory in get_subdirectories(directory):
        artifact_paths = glob(os.path.join(subdirectory, "out*.*"))
        params_path = os.path.join(subdirectory, "params.json")

        with open(params_path) as params_file:
            params = json.load(params_file)

        yield artifact_paths, params


if __name__ == "__main__":
    for x, y in iter_artifact_dir(
        "/home/cbl05/workspace/tiresias/data/interim/adjacency_matrices/multi_layer"
    ):
        print(x, y)
