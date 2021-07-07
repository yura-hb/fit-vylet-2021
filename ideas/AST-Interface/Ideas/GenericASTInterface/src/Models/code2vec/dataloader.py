
import pytorch_lightning as pl
import requests as r

import os
import json
import tarfile as tar

from ...Derived.DummyPythonTree import DummyPythonTree

class PythonASTPathDataloader(pl.LightningDataModule):

  def __init__(self, output_train_file: str, output_test_file: str, should_split: bool):
    """ Initializes a python dataset, where the input is a list of python json ASTs.

    Args:
        output_train_file (str): the processed train file path
        output_test_file (str): the processed test file path
        should_split (bool): If True will split all ASTs to have only at max one class and one function. This will reduce
                             the amount of paths and make them shorter.
    """
    super().__init__()

    self.link = 'http://files.srl.inf.ethz.ch/data/py150.tar.gz'
    self.output_train_file = output_train_file
    self.output_test_file = output_test_file
    self.should_split = should_split

    self.word_index = { 'unk': 0, 'pad': 1 }
    self.path_index = { 'unk': 0, 'pad': 1 }
    self.target_index = { 'unk': 0, 'pad': 1 }

    self.index_word = {}
    self.index_path = {}
    self.index_target = {}


  def prepare_data(self):
    """ The preparation is done in the next steps:
      1.

    """
    request = r.get(self.link, allow_redirects = True)

    # Download data

    with open('python_ast_dataset.tar.gz', 'wb') as f:
      f.write(request.content)

    with tar.open('python_ast_dataset.tar.gz', 'r:gz') as f:
      f.extractall()
    # Process all the ASTs
    # 1. Load the json.
    # 2. Generate a AST and split it on subtrees.
    # 3. Create a dict to encode a tree
    # 4. For each tree generate pathes
    # TODO: - Implement
    pass


  def __generate_paths(self, file: str, output_file: str):
    """ Processes a json of the AST trees and generate paths:
      1. Load a tree
      2. Split it.
      3. Generate paths
      4. Update dictionary
      5. Write to output file

    Args:
        file (str): input file path
        output_file ([type]): output file path
    """

    with open(file) as input, open(output_file, 'wb', 2048) as out:
      for line in input:
        js = json.loads(line)
        tree = DummyPythonTree.from_json(js)

        for tree in tree.split():
          for path in tree.gen_terminal_paths():
            pass


