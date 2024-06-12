import torch.nn as nn
import torch


class CustomBceLossWithWeights(nn.Module):
    def __init__(self, sample_weights, pos_weight):
        super(CustomBceLossWithWeights, self).__init__()
        self.sample_weights = sample_weights
        self.loss_fn_with_pos_weight = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor([pos_weight])
        )
        self.loss_fn_without_pos_weight = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, y_hat, y):
        # check if y has any positive values
        if y.sum() > 0:
            loss = self.loss_fn_with_pos_weight(y_hat, y)
        else:
            loss = self.loss_fn_without_pos_weight(y_hat, y)

        weights = torch.tensor(
            [weight for weight in self.sample_weights],
            dtype=torch.float32,
            device=y.device,
        )
        weighted_loss = loss * weights
        return weighted_loss.mean()
