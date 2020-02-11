import os
from os import path
import yaml
import json
from operator import itemgetter
from types import SimpleNamespace
from src.artifacts import ArtifactStore


root_dir = path.abspath(path.join(path.dirname(__file__), ".."))

temp_dir = path.join(os.sep, "tmp", "tiresias", "ray")

with open(path.join(root_dir, "config.yml"), "r") as stream:
    yaml_config = yaml.safe_load(stream)

# System
cpus, gpus, memory = itemgetter("cpus", "gpus", "memory")(yaml_config["system"])

# Input files
edge_lists_files = yaml_config["data"]["edge_lists"]
node_attributes_file = yaml_config["data"]["node_attributes"]
train_node_labels_file = yaml_config["data"]["train_node_labels"]
test_node_labels_file = yaml_config["data"]["test_node_labels"]




# Enabled models
models = yaml_config["models"]

# Always use absolute paths
edge_lists_files = [
    path.join(root_dir, filename) if not path.isabs(filename) else filename
    for filename in edge_lists_files
]

if not path.isabs(node_attributes_file):
    node_attributes_file = path.join(root_dir, node_attributes_file)

if not path.isabs(train_node_labels_file):
    train_node_labels_file = path.join(root_dir, train_node_labels_file)

if not path.isabs(train_node_labels_file):
    test_node_labels_file = path.join(root_dir, test_node_labels_file)


#Input processed files
train_node_labels_file_processed = train_node_labels_file.replace(".tsv", "_processed.tsv")
test_node_labels_file_processed = test_node_labels_file.replace(".tsv", "_processed.tsv")
edge_lists_files_processed = [f.replace(".tsv", "_processed.tsv") for f in edge_lists_files]
node_attributes_file_processed = node_attributes_file.replace(".tsv", "_processed.tsv")


parameters = dict()

with open(path.join(root_dir, "parameters", "features.json")) as param_file:
    parameters["features"] = json.load(param_file)

with open(path.join(root_dir, "parameters", "models_validation.json")) as param_file:
    parameters["validation"] = json.load(param_file)

with open(path.join(root_dir, "parameters", "models_test.json")) as param_file:
    parameters["test"] = json.load(param_file)

artifact_stores = SimpleNamespace(
    adjacency_matrices=SimpleNamespace(
        merged_layer=ArtifactStore(
            path.join(root_dir, "artifacts", "adjacency_matrices", "merged_layer")
        ),
        multi_layer=ArtifactStore(
            path.join(root_dir, "artifacts", "adjacency_matrices", "multi_layer")
        ),
    ),
    edge_lists=ArtifactStore(path.join(root_dir, "artifacts", "edge_lists")),
    random_walks=ArtifactStore(path.join(root_dir, "artifacts", "random_walks")),
    embeddings=ArtifactStore(path.join(root_dir, "artifacts", "embeddings")),
)

REPORTS_DIR = path.join(root_dir, "reports")
if not path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

NODE2VEC_DOCKER_IMAGE = "thibaudma/node2vec"
