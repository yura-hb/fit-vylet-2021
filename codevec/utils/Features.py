
from dataclasses import dataclass, field, fields
from sys import setdlopenflags

from torch import Tensor
from torch import tensor

from typing import Dict

@dataclass
class Features:
  input_ids: Tensor = field(default = tensor([]))
  attention_mask: Tensor = field(default = tensor([]))
  token_type_ids: Tensor = field(default = tensor([]))

  token_embeddings: Tensor = field(default = tensor([]))
  cls_token: Tensor = field(default = tensor([]))
  hidden_states: Tensor = field(default = tensor([]))

  def trim_input(self):
    self.input_ids = tensor([])

  def trim_output(self):
    self.token_embeddings = tensor([])
    self.cls_token = tensor([])
    self.hidden_states = tensor([])

  def trim_hidden_layers(self):
    self.hidden_states = tensor([])

  @property
  def model_input(self) -> Dict:
    return {
      'input_ids': self.input_ids,
      'attention_mask': self.attention_mask,
      'token_type_ids': self.token_type_ids
    }
