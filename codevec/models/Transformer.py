#
# Thanks for inspiration:
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
#

from transformers import AutoModel, AutoTokenizer, AutoConfig
from pytorch_lightning import LightningModule
from dataclasses import dataclass, field

from typing import (Dict, Union, List, Tuple)

from ..utils import RawFeatures, EmbeddedFeatures

import torch

class Transformer(LightningModule):

  @dataclass
  class Config:

    @dataclass
    class SplitConfig:
      stride: int = 128

    model_name: str
    tokenizer_name: str
    output_hidden_states: bool = False
    split_config: SplitConfig = None
    model_args: Dict = field(default_factory=dict)
    tokenizer_args: Dict = field(default_factory=dict)
    autograd: bool = False
    requires_additional_pad_token: bool = False

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    if self.config.output_hidden_states:
      self.config.model_args['output_hidden_states'] = True

    self.transformer_config = AutoConfig.from_pretrained(config.model_name, **config.model_args)
    self.model = AutoModel.from_pretrained(config.model_name, config=self.transformer_config)
    self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, **config.tokenizer_args)

    if config.requires_additional_pad_token:
      self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  def __repr__(self):
    return "Transformer({}) with Transformer model: {} ".format(self.config, self.model.__class__.__name__)

  def forward(self, x: RawFeatures) -> EmbeddedFeatures:
    """
    Passes input tensor from the model. The input should be tokenized before forward.

    Args:
        x (tensor])
    """
    output = None

    if self.config.autograd:
      output = self.model(**x.model_input, return_dict=True)
    else:
      with torch.no_grad():
        output = self.model(**x.model_input, return_dict=True)

    return EmbeddedFeatures.from_transformer(x, output, self.config.output_hidden_states)

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
      'max_length': min(self.tokenizer.model_max_length, self.transformer_config.max_position_embeddings)
    }

    if self.config.split_config:
      kwargs['return_overflowing_tokens'] = True
      kwargs['stride'] = self.config.split_config.stride
      kwargs['max_length'] = min(self.tokenizer.model_max_length, self.transformer_config.max_position_embeddings)

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

  def __parse_tokenizer_input(self, text: Union[str, List[str]]) -> List[str]:
    """ Parsers tokenizer input

    Args:
        text (Union[str, List[str]]): texts to encode with transformer

    Returns:
        [List[str]]: Output + texts needed for tokenization.
    """

    if isinstance(text, str):
      items = [text]
      return items

    return text
