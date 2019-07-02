# coding: utf-8

import torch
from src.utils.misc import random_choice


class Bagging:
    """Bagging for PU learning."""

    def __init__(self, bagging_model, n_bootstraps=1, **kwargs):
        self.n_bootstraps = n_bootstraps
        self.predictions = None
        self.is_cuda = None

        # Instantiate model
        self.model = bagging_model(**kwargs)

    def cuda(self):
        """Moves all model parameters and buffers to the GPU."""
        self.model.cuda()
        self.is_cuda = True

    def reset_parameters(self):
        self.predictions = None

    def fit(self, train_labels, train_mask):
        device = train_labels.device
        n_nodes = train_mask.size(0)

        # Mark unknown labels as -1
        labels = torch.empty(n_nodes, dtype=torch.long, device=device).fill_(-1)
        labels[train_mask] = train_labels

        # Get indices of zero and one labels that can be used for training
        zero_labels_idx = (labels == 0).nonzero().view(-1)
        one_labels_idx = (labels == 1).nonzero().view(-1)

        n_one_labels = one_labels_idx.size(0)

        # Reset predictions
        self.predictions = torch.zeros(n_nodes, dtype=torch.float, device=device)

        for bootstrap in range(self.n_bootstraps):
            print(f"Bootstrap {bootstrap + 1} / {self.n_bootstraps}")

            # Get a random sample of size nb_pos_genes from zero labels indices
            random_sample_zero_labels_idx = random_choice(zero_labels_idx, n_one_labels)

            # Get a train mask for the current bootstrap
            bootstrap_train_instances_idx = torch.cat(
                (one_labels_idx, random_sample_zero_labels_idx), dim=0
            )

            # Generate train mask from indices
            bootstrap_train_mask = torch.zeros(n_nodes, dtype=torch.uint8, device=device)
            bootstrap_train_mask[bootstrap_train_instances_idx] = 1

            # Reset model parameters
            self.model.reset_parameters()

            # Train model with current bootstrap
            self.model.fit(
                train_labels=labels[bootstrap_train_mask],
                train_mask=bootstrap_train_mask
            )

            # Predict with current bootstrap
            self.predictions += self.model.predict_proba()

        self.predictions /= self.n_bootstraps

    def predict_proba(self):
        return self.predictions
