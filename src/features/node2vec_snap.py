import re
import docker
from src import config

CONTAINER_INPUT_FILE = "/input.txt"
CONTAINER_OUTPUT_FILE = "/output.txt"


def run(
    input_edge_list,
    output_file,
    dimensions=128,
    walk_length=80,
    n_walks=10,
    context_size=10,
    epochs=1,
    p=1,
    q=1,
    verbose=True,
    directed_graph=False,
    weighted_graph=False,
    output_random_walks=False,
):
    """Wrapper around node2vec C++ SNAP implementation.

    Parameters
    ----------
    input_edge_list: str
        Absolute path of the input edge list file.

    output_file: str
        Absolute path of the output file (embedding or random walk file).

    dimensions: int (default: 128)
        Number of dimensions of the generated vector embeddings.

    walk_length: int (default: 80)
        Length of walk per source node.

    n_walks: int (default: 10)
        Number of walks per source node.

    context_size: int (default: 10)
        Context size in Word2Vec.

    epochs: int (default: 1)
        Number of epochs in stochastic gradient descent.

    p: int (default: 1)
        Return hyperparameter.

    q: int (default: 1)
        Inout hyperparameter.

    verbose: bool (default: True)
        Verbosity of the output.

    directed_graph: bool (default: False)
        Indicates whether the graph is directed.

    weighted_graph: bool (default: False)
        Indicates whether the graph is weighted.

    output_random_walks: bool (default: False)
        Output random walks instead of node embeddings.

    References
    ----------
    https://github.com/snap-stanford/snap/blob/master/examples/node2vec/ReadMe.txt
    """
    numeric_params = [
        f"-{key}:{value}"
        for key, value in {
            "i": CONTAINER_INPUT_FILE,
            "o": CONTAINER_OUTPUT_FILE,
            "d": dimensions,
            "l": walk_length,
            "r": n_walks,
            "k": context_size,
            "e": epochs,
            "p": p,
            "q": q,
        }.items()
        if value
    ]

    bool_params = [
        f"-{key}"
        for key, value in {
            "v": verbose,
            "dr": directed_graph,
            "w": weighted_graph,
            "ow": output_random_walks,
        }.items()
        if value
    ]

    cmd_args = " ".join((*numeric_params, *bool_params))
    run_docker_container(input_edge_list, output_file, cmd_args, verbose)

    print("Done!")


def run_docker_container(input_file, output_file, cmd_args, verbose=True):
    client = docker.from_env()

    try:
        touch(output_file)

        container = client.containers.run(
            config.NODE2VEC_DOCKER_IMAGE,
            cmd_args,
            volumes={
                input_file: {"bind": CONTAINER_INPUT_FILE, "mode": "ro"},
                output_file: {"bind": CONTAINER_OUTPUT_FILE, "mode": "rw"},
            },
            detach=True,
        )

        if verbose:
            for line in container.logs(stream=True):
                line = line.strip().decode("utf-8")
                if re.fullmatch(r"[a-zA-Z]+\s[a-zA-Z]+: \d\d?.\d\d%", line):
                    print(line, flush=True, end="\r")
                else:
                    print(line, flush=True)
            print()

        container.wait()
        container.remove()

    finally:
        client.close()


def touch(filepath):
    open(filepath, "a").close()
