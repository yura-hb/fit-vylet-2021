
import pytorch_lightning as pl
import glob
import math

from torch.utils.data import DataLoader, IterableDataset
from dataclasses import dataclass
from typing import List, Optional

from codevec.utils.EmbeddedFeatures import EmbeddedFeatures

class EmbeddedFilesDataset(IterableDataset):

  def __init__(self, files: List[str]):
    super().__init__()

    self.files = files

  def __iter__(self):
    for file in self.files:
      features = EmbeddedFeatures.read(file)
      embeddings = features.token_embeddings[features.attention_mask == True]

      for embedding in embeddings:
        yield embedding


class EmbeddedDirectoryDataModule(pl.LightningDataModule):

  @dataclass
  class Config:
    directory: str

    batch_size: int = 32

    train_files_ratio: float = 0.9
    test_files_ratio: float = 0.05
    validation_file_ratio: float = 0.05

    file_regexes: List[str] = ['*.pt']

  def __init__(self, config: Config):
    super().__init__()

    self.config = config

  def setup(self, stage: Optional[str] = None) -> None:
    files = self.fetch_files(self.config.directory, self.config.file_regexes)

    assert len(files) > 0, "At least one file must match the predicate"
    assert self.config.train_files_ratio + self.config.test_files_ratio + self.config.validation_file_ratio <= 1, \
           "Overall split ratio must be less, than one"

    files_count = len(files)

    train_files_count = math.floor(files_count * self.config.train_files_ratio)
    test_files_count = math.floor(files_count * self.config.test_files_ratio)
    validation_files_count = math.floor(files_count * self.config.validation_file_ratio)

    self.train_files = files[:train_files_count]
    del files[:train_files_count]

    self.test_files = files[:test_files_count]
    del files[:test_files_count]

    self.validation_files = files[:validation_files_count]
    del files[:validation_files_count]

    if len(files) > 0:
      self.train_files += files

  def train_dataloader(self) -> DataLoader:
    dataset = EmbeddedFilesDataset(self.train_files)

    return DataLoader(dataset, batch_size=self.config.batch_size)

  def val_dataloader(self) -> DataLoader:
    dataset = EmbeddedFilesDataset(self.validation_files)

    return DataLoader(dataset, batch_size=self.config.batch_size)

  def test_dataloader(self) -> DataLoader:
    dataset = EmbeddedFilesDataset(self.test_files)

    return DataLoader(dataset, batch_size=self.config.batch_size)

  @staticmethod
  def fetch_files(directory: str, file_regexes: List[str]) -> List[str]:
    filenames = []

    for regex in file_regexes:
      filenames += glob.glob(directory + '/' + regex, recursive=True)

    return filenames

  @staticmethod
  def count_embeddings(filenames):
    print('Begin counting embeddings')

    count = 0

    for filename in filenames:
      embedding = EmbeddedFeatures.read(filename)

      count += embedding