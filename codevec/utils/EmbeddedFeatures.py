from dataclasses import dataclass, field

from torch import Tensor, tensor, cat, save, load, max, unique, sum, zeros

from typing import Dict, List

import os

@dataclass
class EmbeddedFeatures:
  token_embeddings: Tensor = field(default = tensor([])) #[BATCH, IDX, EMBEDDING_LEN]
  cls_token: Tensor = field(default = tensor([])) #[BATCH, EMBEDDING_LENGTH]
  attention_mask: Tensor = field(default = tensor([])) #[BATCH, IDX, EMBEDDING_LEN]
  hidden_states: Tensor = field(default = tensor([])) #[LAYER, BATCH, IDX, EMBEDDING_LEN]
  sample_mapping: Tensor = field(default=tensor([])) # [BATCH]

  def __len__(self):
    return self.token_embeddings.shape[0]

  def merge(self, features):
    self.token_embeddings = cat([self.token_embeddings, features.token_embeddings])
    self.attention_mask = cat([self.attention_mask, features.attention_mask])
    self.cls_token = cat([self.cls_token, features.cls_token])
    self.sample_mapping = cat([self.sample_mapping, features.sample_mapping])
    self.hidden_states = cat([self.hidden_states, features.hidden_states])

  def trim_hidden_layers(self):
    self.hidden_states = tensor([])

  def at(self, indicies: List[int]):
    """
    Creates a copy of the tensor at the specific indicies

    Args:
        indicies [List[int]] - A list of indicies
    """
    tensor_index = tensor(indicies)

    return EmbeddedFeatures(
      token_embeddings = self.token_embeddings.index_select(0, tensor_index) if not self.token_embeddings.numel() == 0 else tensor([]),
      cls_token = self.cls_token.index_select(0, tensor_index) if not self.cls_token.numel() == 0 else tensor([]),
      attention_mask = self.attention_mask.index_select(0, tensor_index) if not self.attention_mask.numel() == 0 else tensor([]),
      hidden_states = self.hidden_states.index_select(0, tensor_index) if not self.hidden_states.numel() == 0 else tensor([]),
      sample_mapping = self.sample_mapping.index_select(0, tensor_index) if not self.sample_mapping.numel() == 0 else tensor([])
    )


  def iterate_samples(self):
    for sample in unique(self.sample_mapping):
      yield EmbeddedFeatures(
        token_embeddings=self.token_embeddings[self.sample_mapping == sample],
        attention_mask=self.attention_mask[self.sample_mapping == sample],
        cls_token=self.cls_token[self.sample_mapping == sample],
        hidden_states=self.hidden_states[self.sample_mapping == sample] if not self.hidden_states.numel() == 0 else tensor([]),
        sample_mapping=zeros(sum(self.sample_mapping[self.sample_mapping == sample]) + sample),
      )

  def to(self, device):
    self.token_embeddings = self.token_embeddings.to(device)
    self.cls_token = self.cls_token.to(device)
    self.attention_mask = self.attention_mask.to(device)
    self.hidden_states = self.hidden_states.to(device)

    return self

  #
  # IO
  #

  @staticmethod
  def read(path: str):
    """ Reads tensor from directory and filename

    Args:
        path(str) A path to read file
    """
    assert os.path.isfile(path), "File should exist"

    d = load(path)

    return EmbeddedFeatures(
      token_embeddings = d['token_embeddings'],
      cls_token = d['cls_token'],
      attention_mask = d['attention_mask'],
      hidden_states = d['hidden_states'],
      sample_mapping=d['sample_mapping']
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
      'hidden_states': self.hidden_states,
      'sample_mapping': self.sample_mapping
    }

    save(d, path)
