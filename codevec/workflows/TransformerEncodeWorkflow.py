from codevec.models import Transformer
from codevec.utils import RawFeatures, EmbeddedFeatures

from dataclasses import dataclass

from typing import Generator, List

import os, shutil, time
import torch

import gc
import pandas as pd

class TransformerEncodeWorkflow:

  @dataclass
  class Config:
    output_dir: str = 'embedded'
    tokens_dir: str = 'tokens'
    embedding_dir: str = 'embedding'
    map_filename: str = 'map.json'
    transformer: Transformer = None
    device: str = 'cpu'
    generate_embedding: bool = False
    tokenize_batch_size: int = 128
    embedding_batch_size: int = 128

  def __init__(self, config, filenames):
    self.config = config
    self.filenames = filenames

  def run(self):
    if os.path.exists(self.config.output_dir):
      shutil.rmtree(self.config.output_dir, ignore_errors=True)

    os.mkdir(self.config.output_dir)
    os.mkdir(self.config.output_dir + '/' + self.config.tokens_dir)

    file_info = self.__tokenize()

    file_info.to_json(path_or_buf=self.config.output_dir + '/' + self.config.map_filename)

    if self.config.generate_embedding:
      os.mkdir(self.config.output_dir + '/' + self.config.embedding_dir)

      self.__gen_embedding(file_info)

  def __tokenize(self):
    """
    Traverses files and write tokenized versions of them to the tokenized directory
    """
    print("Begin Tokenization. Total files {}".format(len(self.filenames)))

    self.output_filenames = set()

    iterator = iter(self.filenames)

    dataframe = pd.DataFrame()
    idx = 0

    for batch in self.__batched(iterator, self.config.tokenize_batch_size):
      print('Start tokenization of batch of {} files'.format(len(batch)))
      start = time.process_time()

      texts = []

      for file in batch:
        with open(file, 'r') as f:
          texts.append(f.read())

      features = self.config.transformer.tokenize(texts)

      for index, feature in enumerate(features.iterate_samples()):
        path = self.__make_tokens_path(idx)

        feature.write(path)
        dataframe = dataframe.append(dict(idx=idx,
                                          filename = batch[index],
                                          blocks = feature.input_ids.shape[0]),
                                          ignore_index=True)
        idx += 1

      end = time.process_time()
      print('End tokenization of batch of {} files and time {}'.format(len(batch), end - start))

    dataframe.set_index('idx')

    return dataframe

  def __gen_embedding(self, file_info):
    print('Begin Embedding generation of {} blocks', file_info['blocks'].sum())

    for batch in self.__batched_features_iterator(file_info, self.config.embedding_batch_size):
      print('Start embedding generation of batch of {} blocks'.format(batch.input_ids.shape[0]))
      start = time.process_time()

      gc.collect()
      torch.cuda.empty_cache()

      encoded = self.config.transformer(batch.to(self.config.device)).to('cpu')

      for embedding in encoded.iterate_samples():
        path = self.__make_embedding_path(int(embedding.sample_mapping[0]))

        if os.path.exists(path):
          features = EmbeddedFeatures.read(path)
          features.merge(embedding)
          features.write(path)
        else:
          embedding.write(path)

      end = time.process_time()
      print('End embedding generation of batch of {} files and time {}'.format(batch.input_ids.shape[0], end - start))

  def __make_tokens_path(self, idx):
    return self.config.output_dir + '/' + self.config.tokens_dir + '/{}.pt'.format(idx)

  def __make_embedding_path(self, idx):
    return self.config.output_dir + '/' + self.config.embedding_dir + '/{}.pt'.format(idx)

  def __batched(self, iterator, count) -> Generator[int, List[str], None]:
    buffer = []

    for element in iterator:
      buffer.append(element)

      if len(buffer) >= count:
        yield buffer
        buffer = []

    if len(buffer) != 0:
      yield buffer

  def __batched_features_iterator(self, file_info: pd.DataFrame, limit: int) -> Generator[int, RawFeatures, None]:
    buffered = RawFeatures()

    for index, row in file_info.iterrows():
      path = self.__make_tokens_path(index)

      features = RawFeatures.read(path)
      features.sample_mapping[True] = index

      # Merge of single file can produce a lot of blocks
      buffered.merge(features)

      if len(buffered) >= limit:
        for i in range(0, len(buffered), limit):
          if i + limit < len(buffered):
            yield buffered.at(list(range(i, i+limit)))
          else:
            buffered = buffered.at(list(range(i, len(buffered))))

    if len(buffered) > 0:
      yield buffered
