from dataclasses import dataclass, field

import torch
from torch import Tensor, tensor, cat, save, load, unique, zeros

from typing import List

import os

@dataclass
class EmbeddedFeatures:
  token_embeddings: Tensor = field(default = tensor([], dtype=torch.float16)) #[BATCH, IDX, EMBEDDING_LEN]
  cls_token: Tensor = field(default = tensor([], dtype=torch.float16)) #[BATCH, EMBEDDING_LENGTH]
  attention_mask: Tensor = field(default = tensor([], dtype=torch.bool)) #[BATCH, IDX, EMBEDDING_LEN]
  hidden_states: Tensor = field(default = tensor([], dtype=torch.float16)) #[LAYER, BATCH, IDX, EMBEDDING_LEN]
  sample_mapping: Tensor = field(default=tensor([], dtype=torch.int)) # [BATCH]

  def __len__(self):
    return self.token_embeddings.shape[0]

  def merge(self, features):
    self.token_embeddings = cat([self.token_embeddings, features.token_embeddings])
    self.attention_mask = cat([self.attention_mask, features.attention_mask])
    self.cls_token = cat([self.cls_token, features.cls_token])
    self.sample_mapping = cat([self.sample_mapping, features.sample_mapping])
    self.hidden_states = cat([self.hidden_states, features.hidden_states], dim = 1)

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
        hidden_states=self.hidden_states[:, self.sample_mapping == sample] if not self.hidden_states.numel() == 0 else tensor([]),
        sample_mapping=zeros(self.sample_mapping[self.sample_mapping == sample].numel()) + sample,
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

    if d.get('embedding_shape'):
      embedding_shape = d['embedding_shape']

      embeddings = d['token_embeddings']
      # (Batch idx * block_length) - actual blocks count
      pad_embedding_count = (embedding_shape[0] * embedding_shape[1]) - embeddings.shape[0]

      if not embedding_shape == embeddings.shape:
        embeddings = cat((embeddings, zeros((pad_embedding_count, embedding_shape[2]))), dim=0)

      # TODO: - Implement hidden states
      return EmbeddedFeatures(
        token_embeddings=embeddings.view(embedding_shape),
        cls_token=d['cls_token'],
        attention_mask=d['attention_mask'],
        hidden_states=d['hidden_states'],
        sample_mapping=d['sample_mapping']
      )

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

    embedding_shape = self.token_embeddings.shape

    # A trick to save a lot of space, because embeddings are in mostly aligned to the size of the block.
    # When a embedding is badly aligned to the size of block we can save n-1 * 768 * 16 bytes of data.

    d = {
      'token_embeddings': self.token_embeddings[self.attention_mask == True],
      'attention_mask': self.attention_mask,
      'hidden_states': self.hidden_states[:, self.attention_mask == True] if self.hidden_states.numel() > 0 else self.hidden_states,
      'cls_token': self.cls_token,
      'sample_mapping': self.sample_mapping,
      'embedding_shape': embedding_shape
    }

    save(d, path)

  @staticmethod
  def from_transformer(raw_features, output):
    states = output[0]

    embedded = EmbeddedFeatures(token_embeddings=states.to(torch.float16),
                                cls_token=states[:, 0, :].to(torch.float16),
                                attention_mask=raw_features.attention_mask.to(torch.bool),
                                sample_mapping=raw_features.sample_mapping.to(torch.int))

    if output.hidden_states:
      embedded.hidden_states = torch.stack(list(output.hidden_states), dim = 0).to(torch.float16)

    return embedded
