from pytorch_lightning import LightningModule
from ..utils.Features import Features

import torch

class MaxPooling(LightningModule):

  def __init__(self, dim: int):
    """
    Args:
        dim (int): A dimension to take max for.
    """
    super().__init__()

    self.dim = dim

  def forward(self, x: Features) -> Features:
    """ Pool by attention mask

    Args:
        x (Features): A features vector

    Returns:
        Features: A features vector, which has gone through max pooling
    """
    mask = x.attention_mask.unsqueeze(-1).expand(x.token_embeddings.size()).float()
    x.token_embeddings[mask == 0] = -1e9
    max_vector = torch.max(x.token_embeddings, dim = self.dim).values

    x.token_embeddings = max_vector
    return x
