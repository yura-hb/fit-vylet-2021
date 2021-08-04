
from codevec.utils import RawFeatures, EmbeddedFeatures

import shutil, os, glob, time

from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Generator

import torch

class BuildIndexWorkflow:
  """
  A simple workflow, which groups embeddings by tokens to analyze specific tokens.
  """

  @dataclass
  class Config:

    embedding_dir: str
    tokens_dir: str
    output_dir: str = 'indexed'

    token_regex: str = "*.pt"
    embedding_regex: str = "*.pt"

    processing_batch_size: int = 32

    tokens: torch.Tensor = field(default = torch.tensor([], dtype=torch.int))

  def __init__(self, config: str):
    self.config = config

  def run(self):
    # Recreate directory
    if os.path.exists(self.config.output_dir):
      shutil.rmtree(self.config.output_dir, ignore_errors=True)

    os.mkdir(self.config.output_dir)

    assert os.path.exists(self.config.tokens_dir), "Tokens must exist in the repository"
    assert os.path.exists(self.config.embedding_dir), "Embeddings must exist in the repository"

    token_files = glob.glob(self.config.tokens_dir + '/' + self.config.token_regex)
    token_files = [file[len(self.config.tokens_dir) + 1:]  for file in token_files]

    iterator = iter(token_files)

    for batch in self.__batched(iterator, self.config.processing_batch_size):
      print('Start embedding generation of batch of {} blocks'.format(len(batch)))
      start = time.process_time()

      self.__build_index(batch)

      end = time.process_time()
      print('End indexing of batch of {} files and time {}'.format(len(batch), end - start))

  def __build_index(self, filenames: List[str]):
    token_filenames = [self.config.tokens_dir + '/' + filename for filename in filenames]
    embedding_filenames = [self.config.embedding_dir + '/' + filename for filename in filenames]

    raw_features = RawFeatures()

    for token_filename in token_filenames:
      raw_features.merge(RawFeatures.read(token_filename))

    embedded_features = EmbeddedFeatures()

    for embedding_filename in embedding_filenames:
      embedded_features.merge(EmbeddedFeatures.read(embedding_filename))

    tokens = self.config.tokens if self.config.tokens.numel() > 0 else torch.unique(raw_features.input_ids)

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
        dir = self.config.output_dir + '/' + str(int(token)) + '/'
        path = dir + timestamp + '.pt'

        os.makedirs(dir, exist_ok=True)

        result.write(path)


  def __batched(self, iterator, count) -> Generator[int, List[str], None]:
    buffer = []

    for element in iterator:
      buffer.append(element)

      if len(buffer) >= count:
        yield buffer
        buffer = []

    if len(buffer) != 0:
      yield buffer