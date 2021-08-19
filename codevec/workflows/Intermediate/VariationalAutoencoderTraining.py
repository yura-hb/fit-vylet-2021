
from codevec.models import VariationalAutoencoder
from codevec.workflows.Base import AnyWorkflow

import pytorch_lightning as pl


from dataclasses import dataclass

class VariationalAutoencoderTrainingWorkflow(AnyWorkflow):
  """
  IN:
  training_dataset: torch.nn.DataSet
  OUT:

  """

  @dataclass
  class Config:
    trainer: pl.Trainer
    model: pl.LightningModule

  def __init__(self, config: Config):
    super().__init__()

    self.config = config

  def run(self):
    datamodule: pl.LightningDataModule = self.get_from_ctx('training_dataset')

    self.config.trainer.fit(self.config.model)



