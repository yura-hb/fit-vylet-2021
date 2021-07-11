#
# Thanks for inspiration:
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
#

from transformers import AutoModel, AutoTokenizer, AutoConfig
from pytorch_lightning import LightningModule
from dataclasses import dataclass, field

from typing import (Dict, Union, List, Tuple)

# TODO: - Implement better handling of the max length in the transformer
# TODO: - Implement better output type for tokenizer/transformer.
#         The current one is pretty dumb and can't autoalign itself

class Transformer(LightningModule):

  @dataclass
  class Config:
    model_name: str
    tokenizer_name: str
    model_args: Dict = field(default_factory = dict)
    tokenizer_args: Dict = field(default_factory= dict)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    self.transformer_config = AutoConfig.from_pretrained(config.model_name, **config.model_args)
    self.model = AutoModel.from_pretrained(config.model_name, config = self.transformer_config)
    self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, **config.tokenizer_args)

  def __repr__(self):
    return "Transformer({}) with Transformer model: {} ".format(self.config, self.model.__class__.__name__)

  def forward(self, x: Dict):
    """
    Passes input tensor from the model. The input should be tokenized before forward.

    Args:
        x (tensor])
    """
    assert isinstance(x, dict) and 'input_ids' in x.keys() and 'attention_mask' in x.keys(), \
           "The input should be a dict with input_ids and attention_mask keys"

    keys = set(['input_ids', 'attention_mask', 'token_type_ids'])
    filtered_x = { key: value for key, value in x.items() if key in keys }

    output = self.model(**filtered_x, return_dict = True)

    # [batch, token, embedding]
    states = output[0]

    # CLS is always a first one.
    cls_token_embedding = states[:, 0, :]

    result = {
      'token_embeddings': states,
      'cls_token': cls_token_embedding,
      'attention_mask': x['attention_mask']
    }

    if self.model.config.output_hidden_states:
      result.update({ 'hidden_states': output.hidden_states })

    return result

  def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
    """ Tokenizes text features

    Args:
        texts (Union[List[str], List[Dict], List[Tuple[str, str]]]): A text to tokenize
    """

    assert len(texts) > 0, "Text should have positive length"

    output, items = self.__parse_tokenizer_input(texts)

    output.update(
      self.tokenizer(*items, padding = True, truncation = 'longest_first', return_tensors = 'pt')
    )

    return output

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
