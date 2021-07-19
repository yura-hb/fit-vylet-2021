
from dataclasses import dataclass, field
from re import M

from torch import Tensor
from torch import tensor
from torch import cat
from torch import save, load

from typing import Dict, List

import os

@dataclass
class RawFeatures:
  input_ids: Tensor = field(default = tensor([]))
  attention_mask: Tensor = field(default = tensor([]))
  token_type_ids: Tensor = field(default = tensor([]))

  def merge(self, features):
    self.input_ids = cat([self.input_ids, features.input_ids])
    self.attention_mask = cat([self.attention_mask, features.attention_mask])
    self.token_type_ids = cat([self.token_type_ids, features.token_type_ids])

  @property
  def model_input(self) -> Dict:
    return {
      'input_ids': self.input_ids,
      'attention_mask': self.attention_mask,
      'token_type_ids': self.token_type_ids
    }

  def to(self, device):
    self.input_ids.to(device)
    self.attention_mask.to(device)
    self.token_type_ids.to(device)

    return self

@dataclass
class EmbeddedFeatures:
  token_embeddings: Tensor = field(default = tensor([]))
  cls_token: Tensor = field(default = tensor([]))
  attention_mask: Tensor = field(default = tensor([]))
  hidden_states: Tensor = field(default = tensor([]))

  def trim_hidden_layers(self):
    self.hidden_states = tensor([])

  def at(self, indicies: List[int]):
    tensor_index = tensor(indicies)

    return EmbeddedFeatures(
      token_embeddings = self.token_embeddings.index_select(0, tensor_index) if not self.token_embeddings.numel() == 0 else tensor([]),
      cls_token = self.cls_token.index_select(0, tensor_index) if not self.cls_token.numel() == 0 else tensor([]),
      attention_mask = self.attention_mask.index_select(0, tensor_index) if not self.attention_mask.numel() == 0 else tensor([]),
      hidden_states = self.hidden_states.index_select(0, tensor_index) if not self.hidden_states.numel() == 0 else tensor([])
    )

  def to(self, device):
    self.token_embeddings.to(device)
    self.cls_token.to(device)
    self.attention_mask.to(device)
    self.hidden_states.to(device)

    return self

  @staticmethod
  def read(path: str):
    """ Reads tensor from directory and filename

    Args:
        directory (str): A directory to read from
        filename (str): A filename to read from
    """
    assert os.path.isfile(path), "File should exist"

    d = load(path)

    return EmbeddedFeatures(
      token_embeddings = d['token_embeddings'],
      cls_token = d['cls_token'],
      attention_mask = d['attention_mask'],
      hidden_states = d['hidden_states']
    )

  def write(self, path: str):
    """Writes all tensors to a file with the filename

    Args:
        path(str): path to write file
    """

    d = {
      'token_embeddings': self.token_embeddings,
      'cls_token': self.cls_token,
      'attention_mask': self.attention_mask,
      'hidden_states': self.hidden_states
    }

    save(d, path)
