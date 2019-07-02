# coding: utf-8

import os
import json
from os import path
from uuid import uuid4


PARAMS_FILENAME = "metadata.json"
ARTIFACT_DIRNAME = "artifacts"


def get_subdirectories(directory):
    subdirectories = next(os.walk(directory))[1]
    return (os.path.join(directory, subdirectory) for subdirectory in subdirectories)


class ArtifactStore:
    def __init__(self, store_path):
        self.store_path = store_path

        if not path.exists(self.store_path):
            os.makedirs(self.store_path)

    def __iter__(self):
        for subdirectory in get_subdirectories(self.store_path):
            artifact_container = ArtifactContainer(subdirectory)
            yield artifact_container.list_artifacts(), artifact_container.load_params()

    def create_artifact_container(self):
        container_path = os.path.abspath(os.path.join(self.store_path, uuid4().hex))
        return ArtifactContainer(container_path)


class ArtifactContainer:
    def __init__(self, container_path):
        self.container_path = container_path
        self.artifact_dir_path = path.join(container_path, ARTIFACT_DIRNAME)
        self.params_path = path.join(container_path, PARAMS_FILENAME)

        if not path.exists(self.artifact_dir_path):
            os.makedirs(self.artifact_dir_path)

    def save_params(self, **params):
        with open(self.params_path, "w") as out_file:
            json.dump(params, out_file, sort_keys=True, indent=2)

    def load_params(self):
        with open(self.params_path, "r") as in_filepath:
            return json.load(in_filepath)

    def create_artifact_filepath(self, filename):
        return path.join(self.artifact_dir_path, filename)

    def list_artifacts(self):
        return [os.path.join(self.artifact_dir_path, filepath) for filepath in os.listdir(self.artifact_dir_path)]
