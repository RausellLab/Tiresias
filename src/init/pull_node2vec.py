# coding: utf-8

from src import config
import subprocess


def pull_node2vec():
    cmd = ["docker", "pull", config.NODE2VEC_DOCKER_IMAGE]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True
    )

    for line in process.stdout:
        print(line.strip())

    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


if __name__ == "__main__":
    pull_node2vec()
