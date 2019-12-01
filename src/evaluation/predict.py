import pandas as pd


def run(labels, model_class, **kwargs):
    """Run test.

    Parameters
    ----------
    labels: torch.LongTensor
        Tensor of target (label) data.

    model_class
        The class of the model on which prediction will be performed.
        It must implement the fit, the predict_proba and the reset_parameters method.

    kwargs
        The arguments used when instantiating the model.

    Returns
    -------
    ranks_df: pd.DataFrame
        A DataFrame containing the rank of the non-labeled nodes. The DataFrame is indexed the number of the nodes.
    """
    use_cuda = labels.is_cuda
    labeled_nodes_mask = labels.byte()

    # Create model
    model = model_class(**kwargs)

    if use_cuda and "cuda" in dir(model):
        model.cuda()

    # Train model
    model.fit(train_labels=labels[labeled_nodes_mask], train_mask=labeled_nodes_mask)

    # Get predictions
    predictions = model.predict_proba()

    if predictions.is_cuda:
        predictions = predictions.cpu()

    predictions_df = pd.DataFrame({"prediction": predictions})
    predictions_df.index.rename("node", inplace=True)

    # Do not take into account known positive genes when computing rank
    if labeled_nodes_mask.is_cuda:
        labeled_nodes_mask = labeled_nodes_mask.cpu()

    non_labeled_mask = ~labeled_nodes_mask
    non_labeled_mask = non_labeled_mask.numpy().astype(bool)
    predictions_df = predictions_df[non_labeled_mask]

    # Compute gene ranks according to predictions
    predictions_df["rank"] = predictions_df.rank(
        ascending=False,
        method="average",  # Equal values are assigned average rank of the group.
    )

    return predictions_df.sort_values(by=["rank"])
