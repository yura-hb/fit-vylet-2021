
from dataclasses import dataclass, field

from torch import Tensor
from torch import tensor
from torch import cat

from typing import Dict, List

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

