#
# Thanks for inspiration:
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
#

from transformers import AutoModel, AutoTokenizer, AutoConfig
from pytorch_lightning import LightningModule
from dataclasses import dataclass, field

from typing import (Dict, Union, List, Tuple)

from ..utils.Features import *

class Transformer(LightningModule):

  @dataclass
  class Config:
    model_name: str
    tokenizer_name: str
    evaluate: bool = True
    output_hidden_states: bool = False
    autocut_model_max_len: bool = False
    model_args: Dict = field(default_factory = dict)
    tokenizer_args: Dict = field(default_factory= dict)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    if self.config.output_hidden_states:
      self.config.model_args['output_hidden_states'] = True

    self.transformer_config = AutoConfig.from_pretrained(config.model_name, **config.model_args)
    self.model = AutoModel.from_pretrained(config.model_name, config = self.transformer_config)
    self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, **config.tokenizer_args)

    if config.evaluate:
      self.model.eval()

  def __repr__(self):
    return "Transformer({}) with Transformer model: {} ".format(self.config, self.model.__class__.__name__)

  def forward(self, x: RawFeatures) -> EmbeddedFeatures:
    """
    Passes input tensor from the model. The input should be tokenized before forward.

    Args:
        x (tensor])
    """

    output = self.model(**x.model_input, return_dict = True)

    # [batch, token, embedding]
    states = output[0]

    embedded = EmbeddedFeatures(states, states[:, 0, :], x.attention_mask)

    if self.config.output_hidden_states:
      embedded.hidden_states = output.hidden_states

    return embedded

  def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]) -> RawFeatures:
    """ Tokenizes text features

    Args:
        texts (Union[List[str], List[Dict], List[Tuple[str, str]]]): A text to tokenize
    """

    assert len(texts) > 0, "Text should have positive length"

    output, items = self.__parse_tokenizer_input(texts)

    kwargs = {
      'padding': True,
      'truncation': 'longest_first',
      'return_tensors': 'pt'
    }

    if self.config.autocut_model_max_len:
      kwargs['max_length'] = self.transformer_config.max_length

    output.update(self.tokenizer(*items, **kwargs))

    features = RawFeatures(
      input_ids = output['input_ids'],
      attention_mask = output['attention_mask'],
      token_type_ids = output['token_type_ids']
    )

    return features

  def __parse_tokenizer_input(self, texts: Union[str, List[str], List[Dict], List[Tuple[str, str]]]):
    """ Parsers tokenizer input

    Args:
        texts (Union[List[str], List[Dict], List[Tuple[str, str]]]): texts to encode with transformer

    Returns:
        [Tuple[Dict, List]]: Output + texts needed for tokenization.
    """
    output = {}
    items = []

    if isinstance(texts[0], str):
      items = [texts]
      return output, items

    if isinstance(texts[0], dict):
      output['text_keys'] = []

      for text in texts:
        key, text_value = next(iter(text.items()))
        items.append(text_value)
        output['text_keys'].append(key)

      items = [items]
      return output, items

    if isinstance(texts[0], tuple):
      first_batch, second_batch = [], []

      for lhs, rhs in texts:
        first_batch.append(lhs)
        second_batch.append(rhs)

      items = [first_batch, second_batch]
      return output, items

    return output, items
