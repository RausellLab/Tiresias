# coding: utf-8

import itertools
import torch


def product_dict(**kwargs):
    """Returns the cartesian product of the parameters as an iterator of dictionaries."""
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def random_choice(a, size):
    """
    Generates a random sample of a given size from a 1-D tensor.
    The sample is drawn without replacement.

    Parameters
    ----------
    a: torch.tensor
        The input tensor from which the sample will be drawn.
    size: int
        The size of the generated sample.

    Returns
    -------
    sample: torch.tensor
        The generated sample.
    """
    permutation = torch.randperm(a.size(0))
    indices = permutation[:size]
    return a[indices]
