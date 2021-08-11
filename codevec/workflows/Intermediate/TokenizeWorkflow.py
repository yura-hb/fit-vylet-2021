from codevec.models import Transformer
from dataclasses import dataclass
from typing import Generator, List, Set, Tuple

import os, shutil, time
import pandas as pd

from ..Base import AnyWorkflow

class TokenizeWorkflow(AnyWorkflow):
  """
  IN: processing_filenames[Set[Str]]
  OUT: tokens_info: pd.DataFrame
  OUT: tokens_dir: str
  """

  @dataclass
  class Config:
    transformer: Transformer

    working_dir: str = 'embedded'
    tokens_dir: str = 'tokens'
    map_filename: str = 'map.json'
    tokenize_batch_size: int = 128

  def __init__(self, config: Config):
    super().__init__()

    self.config = config

  def run(self):
    filenames = self.get_from_ctx('processing_filenames')

    os.makedirs(self.config.working_dir + '/' + self.config.tokens_dir, exist_ok=True)

    file_info = self.__tokenize(self.config, filenames)
    file_info.to_json(path_or_buf=self.config.working_dir + '/' + self.config.map_filename)

    self.update_ctx(dict(tokens_info=file_info,
                         tokens_dir=self.config.working_dir + '/' + self.config.tokens_dir))

  @staticmethod
  def __tokenize(config: Config, filenames: Set[str]) -> pd.DataFrame:
    """
    Traverses files and write tokenized versions of them to the tokenized directory
    """
    print("Begin Tokenization. Total files {}".format(len(filenames)))

    iterator = iter(filenames)

    dataframe = pd.DataFrame()
    idx = 0

    for batch in TokenizeWorkflow.__batched(iterator, config.tokenize_batch_size):
      print('Start tokenization of batch of {} files'.format(len(batch)))
      start = time.process_time()

      texts = []

      for file in batch:
        with open(file, 'r') as f:
          texts.append(f.read())

      features = config.transformer.tokenize(texts)

      for index, feature in enumerate(features.iterate_samples()):
        path = TokenizeWorkflow.__make_tokens_path(idx, config)

        feature.write(path)
        dataframe = dataframe.append(dict(idx=idx,
                                          filename = batch[index],
                                          tokens_path = path,
                                          blocks = feature.input_ids.shape[0]),
                                          ignore_index=True)
        idx += 1

      end = time.process_time()
      print('End tokenization of batch of {} files and time {}'.format(len(batch), end - start))

    dataframe.set_index('idx')

    return dataframe

  @staticmethod
  def __make_tokens_path(idx, config: Config):
    return config.working_dir + '/' + config.tokens_dir + '/{}.pt'.format(idx)

  @staticmethod
  def __batched(iterator, count) -> Generator[int, List[str], None]:
    buffer = []

    for element in iterator:
      buffer.append(element)

      if len(buffer) >= count:
        yield buffer
        buffer = []

    if len(buffer) != 0:
      yield buffer