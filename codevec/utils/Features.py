
from dataclasses import dataclass, field

from torch import Tensor
from torch import tensor
from torch import cat

from typing import Dict

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

  def trim_output(self):
    self.token_embeddings = tensor([])
    self.cls_token = tensor([])
    self.hidden_states = tensor([])

  def trim_hidden_layers(self):
    self.hidden_states = tensor([])
