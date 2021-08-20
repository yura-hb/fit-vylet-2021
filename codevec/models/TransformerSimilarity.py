
from pytorch_lightning import LightningModule
from dataclasses import dataclass

from codevec.models import Transformer

from typing import Any
from codevec.utils import RawFeatures

import torch

class TransformerSimilarity(LightningModule):
  """
  A transformer model for fine-tuning with pytorch
  """
  @dataclass
  class Config:
    transformer: Transformer
    learning_rate: float = 0.002

  def __init__(self, config):
    super().__init__()

    self.config = config
    self.dropout = torch.nn.Dropout(config.transformer.model.config.hidden_dropout_prod)
    self.classifier = torch.nn.Linear(3 * config.hidden_size, 1)
    self.loss = torch.nn.MSELoss()
    self.learning_rate = config.learning_rate

  def forward(self, input: RawFeatures) -> Any:
    """
    Args:
      input: A raw features containing two blocks of data

    Returns: A similarity value
    """
    assert input.input_ids[0] == 2, "Only two blocks can be processed with the similarity"

    embedded_features = self.config.transformer(input)

    output = embedded_features.cls_token
    lhs, rhs = tuple(output)

    lhs_pooled, rhs_pooled = self.dropout(lhs), self.dropout(rhs)
    vector = torch.cat([lhs_pooled, rhs_pooled, torch.abs(lhs_pooled - rhs_pooled)])

    return self.classifier(vector)

  def training_step(self, batch, batch_idx):
    x, y = batch

    prediction = self(x)

    loss = self.loss(y, prediction)

    self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return loss

  def configure_optimizers(self):
    # (self.lr or self.config.learning_rate) enables automatic lr finding from paper
    # https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html
    return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))

  def predict_step(self, batch: Any, batch_idx: int) -> Any:
    x, _ = batch

    return self(x)