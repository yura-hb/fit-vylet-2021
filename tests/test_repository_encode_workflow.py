
import pytest

from codevec.workflows import TransformerEncodeWorkflow
from codevec.workflows import RepositoryEncodeWorkflow
from codevec.models import Transformer
from codevec.utils import RawFeatures

import os
import glob
import shutil
import pandas as pd

class TestRepositoryEncodeWorkflow:

  output_dir: str = 'test_output'
  cache_dir: str = output_dir + '/cache_dir/'

  def setup(self):
    if not os.path.exists(self.output_dir):
      os.mkdir(self.output_dir)

  def test_tokenization(self):
    wd = os.getcwd()

    filenames, config, map_path, tokens_dir_path, _ = self.tokenize(False)

    _filenames = set()

    for regex in config.file_regexes:
      _filenames.update(glob.glob(regex, recursive=True))

    assert filenames == _filenames

    dataframe = pd.read_json(map_path)

    for index, row in dataframe.iterrows():
      path = tokens_dir_path + '/{}.pt'.format(index)

      features = RawFeatures.read(path)

      assert len(features) == row['blocks'], 'Amount of blocks should be the same as in file'

    os.chdir(wd)

  #@pytest.mark.skip(reason="Extremely long time on CPU, enable GPU to test this")
  def test_embedding_generation(self):
    """
    Warning: Test only on GPU, tests on CPU will take extremely long time to complete
    """
    wd = os.getcwd()

    self.tokenize(True)

    os.chdir(wd)
    assert False
    pass

  def tokenize(self, generate_embedding: bool):
    split_config = Transformer.Config.SplitConfig(128)
    config = Transformer.Config('bert-base-cased',
                                'bert-base-cased',
                                output_hidden_states=False,
                                split_config=split_config,
                                model_args={'output_hidden_states': False},
                                autograd=False)
    bert_model = Transformer(config)

    transformer_workflow_config = TransformerEncodeWorkflow.Config(transformer=bert_model,
                                                                   generate_embedding=generate_embedding,
                                                                   embedding_batch_size=1)

    config = RepositoryEncodeWorkflow.Config(repository_url='git@github.com:iluwatar/java-design-patterns.git',
                                             output_dir=self.cache_dir,
                                             file_regexes=['java-design-patterns/**/*.java'],
                                             transformer_workflow_config=transformer_workflow_config)
    workflow = RepositoryEncodeWorkflow(config)

    workflow.run()

    os.chdir(self.cache_dir)

    map_path = os.getcwd() + '/' + transformer_workflow_config.output_dir + '/' + transformer_workflow_config.map_filename
    tokens_dir_path = transformer_workflow_config.output_dir + '/' + transformer_workflow_config.tokens_dir
    embedding_path = transformer_workflow_config.output_dir + '/' + transformer_workflow_config.embedding_dir

    assert os.path.exists(map_path)
    assert os.path.exists(tokens_dir_path)

    if generate_embedding:
      assert os.path.exists(embedding_path)

    return workflow._filenames, config, map_path, tokens_dir_path, embedding_path
