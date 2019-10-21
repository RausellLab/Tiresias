import torch
import pandas as pd
import numpy as np


def run(labels, train_mask, model_class, **kwargs):
    """
    Run test.

    Parameters
    ----------
    labels: torch.LongTensor
        Tensor of target (label) data.

    train_mask: torch.ByteTensor
        Boolean mask of size n_nodes indicating the nodes used in training.

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

    test_mask = ~train_mask
    n_nodes = labels.size(0)

    # Create model
    model = model_class(**kwargs)

    if use_cuda and "cuda" in dir(model):
        model.cuda()

    # Train model
    model.fit(train_labels=labels[train_mask], train_mask=train_mask)

    # Get predictions
    predictions = model.predict_proba()

    if predictions.is_cuda:
        predictions = predictions.cpu()

    train_labels = torch.zeros(n_nodes, dtype=torch.long, device=device)
    train_labels[train_mask] = labels[train_mask]

    test_labels = torch.zeros(n_nodes, dtype=torch.long, device=device)
    test_labels[test_mask] = labels[test_mask]

    train_pos_node_indices = train_labels.nonzero().squeeze(1)
    test_pos_node_indices = test_labels.nonzero().squeeze(1)

    if train_pos_node_indices.is_cuda:
        train_pos_node_indices = test_pos_node_indices.cpu()

    if test_pos_node_indices.is_cuda:
        test_pos_node_indices = test_pos_node_indices.cpu()

    train_pos_node_indices = train_pos_node_indices.numpy()
    test_pos_node_indices = test_pos_node_indices.numpy()

    # Create a DataFrame to store ranks
    ranks_df = pd.DataFrame(
        data=np.zeros(test_pos_node_indices.size, dtype=int),
        index=test_pos_node_indices,
        columns=["rank"],
    )
    ranks_df.index.name = "pos_node_index"

    for i, current_node_idx in enumerate(test_pos_node_indices):
        predictions_df = pd.DataFrame({"prediction": predictions})

        # Do not take into account known positive genes (except current gene) when computing rank
        mask = np.ones(n_nodes, dtype=bool)
        mask[test_pos_node_indices] = False  # mask other test nodes
        mask[train_pos_node_indices] = False  # mask train test nodes
        mask[current_node_idx] = True
        predictions_df = predictions_df[mask]

        # Compute gene ranks according to predictions
        predictions_df["rank"] = predictions_df.rank(
            ascending=False,
            method="max",  # Equal values are assigned the highest rank in the group i.e. the worst.
        )

        # Add rank of current gene to dataframe
        current_gene_rank = predictions_df.at[current_node_idx, "rank"]
        ranks_df.at[current_node_idx, "rank"] = current_gene_rank

    return ranks_df.sort_index()
