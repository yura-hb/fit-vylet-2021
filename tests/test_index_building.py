import pytest, os, glob
from codevec.workflows import BuildIndexWorkflow

class TestIndexBuilding:

  source_dir = 'output/embedded/'
  tokens_dir = source_dir + 'tokens'
  embedding_dir = source_dir + 'embedding'

  def test_index_buiding(self):
    if not os.path.exists(self.embedding_dir) and not os.path.exists(self.tokens_dir):
      pytest.skip('Please update configuration or place correctly embedding directory')

    config = BuildIndexWorkflow.Config(embedding_dir=self.embedding_dir,
                                       tokens_dir=self.tokens_dir,
                                       output_dir=self.source_dir + 'index')

    workflow = BuildIndexWorkflow(config)

    workflow.run()

    filenames = glob.glob(config.tokens_dir + '/' + config.token_regex)