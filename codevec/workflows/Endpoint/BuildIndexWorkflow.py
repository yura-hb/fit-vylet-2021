from codevec.utils import RawFeatures, EmbeddedFeatures

import shutil, os, glob, time

from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Generator

import torch

from ..Base import AnyWorkflow

class BuildIndexWorkflow(AnyWorkflow):
  """
  IN: tokens_dir[str]
  IN: embedding_dir[str]
  OUT: None
  """

  @dataclass
  class Config:
    working_dir: str = 'embedded'
    output_dir: str = 'indexed'

    token_regex: str = "*.pt"
    embedding_regex: str = "*.pt"

    processing_batch_size: int = 32

    tokens: torch.Tensor = field(default = torch.tensor([], dtype=torch.int))

  def __init__(self, config: Config):
    super().__init__()

    self.config = config

  def run(self):
    tokens_dir = self.get_from_ctx('tokens_dir')
    embedding_dir = self.get_from_ctx('embedding_dir')

    token_files = glob.glob(tokens_dir + '/' + self.config.token_regex)
    token_files = [file[len(tokens_dir) + 1:]  for file in token_files]

    iterator = iter(token_files)

    for batch in self.__batched(iterator, self.config.processing_batch_size):
      print('Start indexing of batch of {} files'.format(len(batch)))
      start = time.process_time()

      self.__build_index(tokens_dir, embedding_dir, batch)

      end = time.process_time()
      print('End indexing of batch of {} files and time {}'.format(len(batch), end - start))

  @staticmethod
  def __build_index(tokens_dir: str, embedding_dir: str, filenames: List[str], config: Config):
    token_filenames = [tokens_dir + '/' + filename for filename in filenames]
    embedding_filenames = [embedding_dir + '/' + filename for filename in filenames]

    raw_features = RawFeatures()

    for token_filename in token_filenames:
      raw_features.merge(RawFeatures.read(token_filename))

    embedded_features = EmbeddedFeatures()

    for embedding_filename in embedding_filenames:
      embedded_features.merge(EmbeddedFeatures.read(embedding_filename))

    tokens = config.tokens if config.tokens.numel() > 0 else torch.unique(raw_features.input_ids)

    for token in tokens:
      if token > 0:
        mask = torch.logical_and(raw_features.input_ids == token, raw_features.attention_mask == 1)

        embedding = embedded_features.token_embeddings[mask]
        selected_batches = torch.sum(mask, dim = 1) > 0

        sample_mapping = embedded_features.sample_mapping[selected_batches]

        result = EmbeddedFeatures(token_embeddings=embedding,
                                  attention_mask=torch.ones(embedding.shape[0]),
                                  sample_mapping=sample_mapping)

        now = datetime.now()
        timestamp = str(now.strftime("%Y%m%d_%H:%M:%S"))
        dir = config.working_dir + '/' + str(int(token)) + '/'
        path = dir + timestamp + '.pt'

        os.makedirs(dir, exist_ok=True)

        result.write(path)

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