from codevec.models import Transformer
from codevec.utils import RawFeatures, EmbeddedFeatures

from dataclasses import dataclass

from typing import Generator

import os, time
import torch

import gc
import pandas as pd

from ..Base import AnyWorkflow

class GenerateEmbeddingWorkflow(AnyWorkflow):
  """
  A workflow, which generates embeddings from raw tokens

  :param[in][ctx] tokens_info:
  :param[in][ctx] tokens_dir:
  :param[out][ctx] embedding_dir:
  """

  @dataclass
  class Config:
    transformer: Transformer

    working_dir: str = 'embedded'
    embedding_dir: str = 'embedding'
    device: str = 'cpu'
    embedding_batch_size: int = 128

  def __init__(self, config: Config):
    super().__init__()

    self.config = config

  def run(self):
    tokens_info = self.get_from_ctx('tokens_info')
    tokens_dir = self.get_from_ctx('tokens_dir')

    os.makedirs(self.config.working_dir + '/' + self.config.embedding_dir, exist_ok=True)

    self.__gen_embedding(tokens_dir, tokens_info, self.config)

    self.update_ctx(dict(embedding_dir = self.config.working_dir + '/' + self.config.embedding_dir))

  @staticmethod
  def __gen_embedding(tokens_dir: str, file_info: pd.DataFrame, config: Config):
    print('Begin Embedding generation of {} blocks', file_info['blocks'].sum())

    for batch in GenerateEmbeddingWorkflow.__batched_features_iterator(tokens_dir, file_info, config.embedding_batch_size):
      print('Start embedding generation of batch of {} blocks'.format(batch.input_ids.shape[0]))
      start = time.process_time()

      gc.collect()
      torch.cuda.empty_cache()

      encoded = config.transformer(batch.to(config.device)).to('cpu')

      for embedding in encoded.iterate_samples():
        path = GenerateEmbeddingWorkflow.__make_embedding_path(config, int(embedding.sample_mapping[0]))

        if os.path.exists(path):
          features = EmbeddedFeatures.read(path)
          features.merge(embedding)
          features.write(path)
        else:
          embedding.write(path)

      end = time.process_time()
      print('End embedding generation of batch of {} files and time {}'.format(batch.input_ids.shape[0], end - start))

  @staticmethod
  def __make_embedding_path(config: Config, idx):
    return config.working_dir + '/' + config.embedding_dir + '/{}.pt'.format(idx)

  @staticmethod
  def __batched_features_iterator(tokens_dir: str, file_info: pd.DataFrame, limit: int) -> Generator[int, RawFeatures, None]:
    buffered = RawFeatures()

    for index, row in file_info.iterrows():
      path = row.tokens_path

      features = RawFeatures.read(path)
      features.sample_mapping[:] = index

      # Merge of single file can produce a lot of blocks
      buffered.merge(features)

      if len(buffered) >= limit:
        for i in range(0, len(buffered), limit):
          if i + limit < len(buffered):
            yield buffered.at(list(range(i, i + limit)))
          else:
            buffered = buffered.at(list(range(i, len(buffered))))

    if len(buffered) > 0:
      yield buffered