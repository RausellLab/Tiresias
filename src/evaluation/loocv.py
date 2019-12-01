import torch
import pandas as pd
import numpy as np


def run(labels, model_class, **kwargs):
    """Run Leave One Out Cross Validation (LOOCV).

    Parameters
    ----------
    labels: torch.LongTensor
        Tensor of target (label) data.

    model_class
        The class of the model on which LOOCV will be performed.
        It must implement the fit, the predict_proba and the reset_parameters method.

    kwargs
        The arguments used when instantiating the model.

    Returns
    -------
    ranks_df: pd.DataFrame
        A DataFrame containing the rank of the positively labelled genes. The DataFrame is indexed by its position in
        the input labels list.
    """
    use_cuda = labels.is_cuda
    device = labels.device

    n_nodes = labels.size(0)
    pos_node_indices = labels.nonzero().squeeze(1)

    if pos_node_indices.is_cuda:
        pos_node_indices = pos_node_indices.cpu()

    pos_node_indices = pos_node_indices.numpy()

    # Create a DataFrame to store ranks across iterations
    ranks_df = pd.DataFrame(
        data=np.zeros(pos_node_indices.size, dtype=int),
        index=pos_node_indices,
        columns=["rank"],
    )
    ranks_df.index.name = "pos_node_index"

    # Create model
    model = model_class(**kwargs)

    if use_cuda and "cuda" in dir(model):
        model.cuda()

    for i, current_node_idx in enumerate(pos_node_indices):
        print(f"LOOCV: iteration {i} / {pos_node_indices.size}")

        train_mask = torch.ones(
            n_nodes, dtype=torch.uint8, device=device
        )  # consider all the instances for training
        train_mask[current_node_idx] = 0  # remove current gene from train_mask
        model.reset_parameters()  # reset model parameters
        model.fit(
            train_labels=labels[train_mask], train_mask=train_mask
        )  # train the model
        predictions = model.predict_proba()  # get predictions

        if predictions.is_cuda:
            predictions = predictions.cpu()

        predictions_df = pd.DataFrame({"prediction": predictions})

        # Do not take into account known positive genes (except current gene) when computing rank
        mask = np.ones(n_nodes, dtype=bool)
        mask[pos_node_indices] = False
        mask[current_node_idx] = True
        predictions_df = predictions_df[mask]

        # Compute gene ranks according to predictions
        predictions_df["rank"] = predictions_df.rank(
            ascending=False,
            # Equal values are assigned the highest rank in the group i.e. the worst.
            method="max",
        )

        # Add rank of current gene to dataframe
        current_gene_rank = predictions_df.at[current_node_idx, "rank"]
        ranks_df.at[current_node_idx, "rank"] = current_gene_rank

    return ranks_df.sort_index()
