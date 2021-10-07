
from transformers import AutoModel, AutoTokenizer, AutoConfig
from pytorch_lightning import LightningModule
from dataclasses import dataclass, field

from typing import (Dict, Union, List, Tuple)

from ..utils import RawFeatures, EmbeddedFeatures

import torch

class Transformer(LightningModule):

  @dataclass
  class SplitConfig:
    stride: int = 128

  @dataclass
  class AutoConfig:
    model_name: str
    tokenizer_name: str
    cache_path: str = None
    model_args: Dict = field(default_factory=dict)
    tokenizer_args: Dict = field(default_factory=dict)

  @dataclass
  class ActionConfig:
    output_hidden_states: bool = False
    autograd: bool = False
    is_gpt_like: bool = False

  def __init__(self, model_config, model, tokenizer, action_config: ActionConfig, split_config: SplitConfig = None):
    super().__init__()

    self.model_config = model_config
    self.model = model
    self.tokenizer = tokenizer

    self.split_config = split_config
    self.action_config = action_config

    self.model.output_hidden_states = action_config.output_hidden_states

    # A help config to allow GPT-like models to correctly process multiple batches of data
    if action_config.is_gpt_like:
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.model.config.pad_token_id = self.model.config.eos_token_id

  def __repr__(self):
    return "Transformer with Transformer model: {} ".format(self.model.__class__.__name__)

  @staticmethod
  def auto_model(auto_config: AutoConfig,
                 action_config: ActionConfig,
                 split_config: SplitConfig = None):
    """
    Creates a model using hugging face auto module

    Args:
      auto_config: Configuration for automodel
      action_config: Configuration for model inference
      split_config: Configuration for split

    Returns: A Transformer object

    """

    params = {}

    if auto_config.cache_path:
      params.update(dict(cache_path = auto_config.cache_path))

    model_config = AutoConfig.from_pretrained(auto_config.model_name, **auto_config.model_args, **params)
    model = AutoModel.from_pretrained(auto_config.model_name, config=model_config)
    tokenizer = AutoTokenizer.from_pretrained(auto_config.tokenizer_name, **auto_config.tokenizer_args)

    return Transformer(model_config, model, tokenizer, action_config, split_config)

  def forward(self, x: RawFeatures) -> EmbeddedFeatures:
    """
    Passes input tensor from the model. The input should be tokenized before forward.

    Args:
        x (tensor])
    """
    output = None

    if self.action_config.autograd:
      output = self.model(**x.model_input,
                          return_dict=True,
                          output_hidden_states = self.action_config.output_hidden_states)
    else:
      with torch.no_grad():
        output = self.model(**x.model_input,
                            return_dict=True,
                            output_hidden_states = self.action_config.output_hidden_states)

    return EmbeddedFeatures.from_transformer(x, output)

  def tokenize(self, texts: Union[str, List[str]]) -> RawFeatures:
    """ Tokenizes text features

    Args:
        texts (Union[List[str], List[Dict], List[Tuple[str, str]]]): A text to tokenize
    """

    assert len(texts) > 0, "Text should have positive length"

    items = self.__parse_tokenizer_input(texts)

    kwargs = {
      'padding': True,
      'truncation': 'longest_first',
      'return_tensors': 'pt',
      'max_length': min(self.tokenizer.model_max_length, self.model_config.max_position_embeddings)
    }

    if self.split_config:
      kwargs['return_overflowing_tokens'] = True
      kwargs['stride'] = self.split_config.stride

    output = self.tokenizer(items, **kwargs)

    features = RawFeatures(
      input_ids=output['input_ids'].to(torch.int),
      attention_mask=output['attention_mask'].to(torch.bool),
      token_type_ids=output.get('token_type_ids').to(torch.int8) if 'token_type_ids' in output.keys() else torch.tensor([]),
      sample_mapping=torch.tensor([])
    )

    overflow_mapping = torch.zeros(features.input_ids.shape[0]) if output.get('overflow_to_sample_mapping') is None \
                       else output.get('overflow_to_sample_mapping')

    features.sample_mapping = overflow_mapping

    return features

  @staticmethod
  def __parse_tokenizer_input(text: Union[str, List[str]]) -> List[str]:
    if isinstance(text, str):
      items = [text]
      return items

    return text
