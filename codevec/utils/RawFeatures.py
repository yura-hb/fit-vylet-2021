
from dataclasses import dataclass, field

import torch
from torch import Tensor, tensor, cat, save, load, max, unique, sum, zeros
from typing import Dict, List

import os

@dataclass
class RawFeatures:
  input_ids: Tensor = field(default = tensor([], dtype=torch.int)) # [BATCH, IDX]
  attention_mask: Tensor = field(default = tensor([], dtype=torch.int)) #[BATCH, [0,1]]
  token_type_ids: Tensor = field(default = tensor([], dtype=torch.int)) #[BATCH, TYPE]
  sample_mapping: Tensor = field(default = tensor([], dtype=torch.int)) # [BATCH]

  def __len__(self):
    return self.input_ids.shape[0]

  def merge(self, features, pad_samples: bool = False):
    """

    Args:
        features(RawFeatures): A features to merge with
        pad_samples(bool): in case if true, will increase sample_mapping in features by max value + 1 in current mapping

    """
    self.input_ids = cat([self.input_ids, features.input_ids]).to(torch.int)
    self.attention_mask = cat([self.attention_mask, features.attention_mask]).to(torch.int)
    self.token_type_ids = cat([self.token_type_ids, features.token_type_ids]).to(torch.int)

    if pad_samples:
      self.sample_mapping = cat([self.sample_mapping,
                                 features.sample_mapping + (max(self.sample_mapping) + 1)]).to(torch.int)
    else:
      self.sample_mapping = cat([self.sample_mapping, features.sample_mapping]).to(torch.int)

  @property
  def model_input(self) -> Dict:
    input = {
      'input_ids': self.input_ids.to(torch.int32),
      'attention_mask': self.attention_mask.to(torch.int32),
    }

    if self.token_type_ids.numel() > 0:
      input.update({ 'token_type_ids': self.token_type_ids.to(torch.int32) })

    return input

  def to(self, device):
    self.input_ids = self.input_ids.to(device)
    self.attention_mask = self.attention_mask.to(device)

    if self.token_type_ids.numel() > 0:
      self.token_type_ids = self.token_type_ids.to(device)

    return self

  def iterate_samples(self):
    for sample in unique(self.sample_mapping):
      yield RawFeatures(
        input_ids=self.input_ids[self.sample_mapping == sample],
        attention_mask=self.attention_mask[self.sample_mapping == sample],
        token_type_ids=self.token_type_ids[self.sample_mapping == sample] if self.token_type_ids.numel() > 0 else tensor([]),
        sample_mapping=zeros(self.sample_mapping[self.sample_mapping == sample].numel()) + sample)


  def at(self, indicies: List[int]):
    """
    Creates a copy of the tensor at the specific indicies

    Args:
      indicies [List[int]] - A list of indicies
    """
    tensor_index = tensor(indicies)

    return RawFeatures(
      input_ids=self.input_ids.index_select(0, tensor_index) if not self.input_ids.numel() == 0 else tensor([]),
      attention_mask=self.attention_mask.index_select(0, tensor_index) if not self.attention_mask.numel() == 0 else tensor([]),
      token_type_ids=self.token_type_ids.index_select(0, tensor_index) if not self.token_type_ids.numel() == 0 else tensor([]),
      sample_mapping=self.sample_mapping.index_select(0, tensor_index) if not self.sample_mapping.numel() == 0 else tensor([])
    )

  #
  # IO
  #

  @staticmethod
  def read(path: str):
    """ Reads tensor from directory and filename

    Args:
        directory (str): A directory to read from
        filename (str): A filename to read from
    """
    assert os.path.isfile(path), "File should exist"

    d = load(path)

    return RawFeatures(
      input_ids=d['input_ids'].to(torch.int),
      attention_mask=d['attention_mask'].to(torch.int),
      token_type_ids=d.get('token_type_ids').to(torch.int) if 'token_type_ids' in d.keys() else tensor([]),
      sample_mapping=d.get('overflow_to_sample_mapping').to(torch.int) if 'overflow_to_sample_mapping' in d.keys() else tensor([])
    )

  def write(self, path: str):
    """Writes all tensors to a file with the filename

    Args:
        path(str): path to write file
    """

    d = {
      'input_ids': self.input_ids,
      'attention_mask': self.attention_mask,
      'token_type_ids': self.token_type_ids,
      'overflow_to_sample_mapping': self.sample_mapping
    }

    save(d, path)
