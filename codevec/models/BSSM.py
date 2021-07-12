
#
# A method of the vector interaction based on the BSSM paper.
#

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule

from .MaxPooling import MaxPooling
from ..utils import EmbeddedFeatures

class BSSM(LightningModule):

  def __init__(self):
    super().__init__()

    self.pooling = MaxPooling(dim = 2)

  def forward(self, features: EmbeddedFeatures):
    """ A model will compute the interaction vector based on the paper.

    1. Calculate weighed softmax sum
    2. Pool items from the vectors
    3. Concat to final result

    Args:
        x (tensor): A tensor to perform operations on. Must have shape [x, z], where z is embedding length
        y (tensor): A tensor to perform operations on. Must have shape [x, z], where z is embedding length
    """
    assert len(features.token_embeddings.shape) == 3 and features.token_embeddings.shape[0] == 2, \
           "Input must have 2 embeddings"

    # [x, z]
    x = features.token_embeddings[0]
    y = features.token_embeddings[1]

    # [z, x] @ [x, z] = [z, z]
    attention = x.T @ y

    # [z, z]
    softmax_x = F.softmax(attention, dim = 1) # By row
    softmax_y = F.softmax(attention, dim = 0) # By column

    # [z, z]
    weighted_x = (softmax_x * torch.sum(y, dim = 0))[:x.shape[0]]
    weighted_y = (softmax_y * torch.sum(x, dim = 0))[:y.shape[0]]

    # [x]
    pooled = self.pooling(features)
    pooled_x, pooled_y = pooled.token_embeddings[0], pooled.token_embeddings[1]

    features.token_embeddings = torch.stack([weighted_x, weighted_y])

    pooled = self.pooling(features)
    pooled_weighted_x, pooled_weighted_y = pooled.token_embeddings[0], pooled.token_embeddings[1]

    # [3 * x]
    concatenated_x = torch.cat([pooled_x, pooled_weighted_x, pooled_x - pooled_weighted_x])
    concatenated_y = torch.cat([pooled_y, pooled_weighted_y, pooled_y - pooled_weighted_y])

    # [9 * x]
    result = torch.cat([concatenated_x, concatenated_y, torch.abs(concatenated_x - concatenated_y)])

    return result

