
#
# A method of the vector interaction based on the BSSM paper.
#
#

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule

class BSSM(LightningModule):

  def __init__(self):
    super().__init__()

  def forward(self, x, y):
    """ A model will compute the interaction vector based on the paper.

    1. Calculate weighed softmax sum
    2. Pool items from the vectors
    3. Concat to final result

    Args:
        x (tensor): A tensor to perform operations on. Must have shape [x, y]
        y (tensor): A tensor to perform operations on. Must have shape [x, y]
    """
    assert x.shape == y.shape, "Embeddings must have the same shape."

    # [y, x] @ [x, y] = [y, y]
    attention = x.T @ y

    # [y, y]
    softmax_x = F.softmax(attention, dim = 1) # By row
    softmax_y = F.softmax(attention, dim = 0) # By column

    # [y, y]
    weighted_x = (softmax_x * torch.sum(y, dim = 0))[:x.shape[0]]
    weighted_y = (softmax_y * torch.sum(x, dim = 0))[:y.shape[0]]

    pool = nn.MaxPool1d(x.shape[1])

    # [x]
    pooled_x = pool(x.view(1, *x.shape)).view(-1)
    pooled_y = pool(y.view(1, *y.shape)).view(-1)
    pooled_weighted_x = pool(weighted_x.view(1, *weighted_x.shape)).view(-1)
    pooled_weighted_y = pool(weighted_y.view(1, *weighted_y.shape)).view(-1)

    # [3 * x]
    concatenated_x = torch.cat([pooled_x, pooled_weighted_x, pooled_x - pooled_weighted_x])
    concatenated_y = torch.cat([pooled_y, pooled_weighted_y, pooled_y - pooled_weighted_y])

    # [9 * x]
    result = torch.cat([concatenated_x, concatenated_y, torch.abs(concatenated_x - concatenated_y)])

    return result

