import numpy as np
from gensim.models import Word2Vec
from src.utils import io


def run(
    random_walk_files, output_file, dimensions=128, context_size=10, epochs=1, workers=1
):
    """Generates node vector embeddings from a list of files containing random
    walks performed on different layers of a multilayer network.

    Parameters
    ----------
    random_walk_files: list
        List of files containing random walks. Each file should correspond to random walks perform on a different layer
        of the network of interest.

    output_file: str
        The file in which the node embeddings will be saved.

    dimensions: int (default: 128)
        Number of dimensions of the generated vector embeddings.

    context_size: int (default: 10)
        Context size in Word2Vec.

    epochs: int (default: 1)
        Number of epochs in stochastic gradient descent.

    workers: int (default: 1)
        Number of worker threads used to train the model.
    """

    walks = np.concatenate([io.read_random_walks(file) for file in random_walk_files])
    walks = walks.astype(str).tolist()

    model = Word2Vec(
        walks,
        size=dimensions,
        window=context_size,
        min_count=0,
        sg=1,  # use skip-gram
        workers=workers,
        iter=epochs,
    )

    model.wv.save_word2vec_format(output_file)
