
from codevec.workflows.TransformerEncodeWorkflow import TransformerEncodeWorkflow
from dataclasses import dataclass

import os
import subprocess
import glob
from typing import List

import shutil

class RepositoryEncodeWorkflow:
  """
  A helper class to fetch/tokenize/embedded git repository
  """

  @dataclass
  class Config:
    repository_url: str
    file_regexes: List[str]
    transformer_workflow_config: TransformerEncodeWorkflow.Config

    output_dir: str = 'cache_dir'

  def __init__(self, config):
    super().__init__()

    self.config = config

  def run(self):
    # Recreate directory
    if os.path.exists(self.config.output_dir):
      shutil.rmtree(self.config.output_dir, ignore_errors=True)

    os.mkdir(self.config.output_dir)

    wd = os.getcwd()

    os.chdir(self.config.output_dir)

    try:
      self.__clone()
      self.__make_index()

      transformer_workflow = TransformerEncodeWorkflow(self.config.transformer_workflow_config,
                                                       filenames=self._filenames)

      transformer_workflow.run()

      os.chdir(wd)
    except Exception as exc:
      os.chdir(wd)

      raise exc

  #
  # Utils
  #

  def __clone(self) -> None:
    # Call Git to clone repository
    process = subprocess.Popen(['git', 'clone', '--depth', '1', '--single-branch', self.config.repository_url],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    process.communicate()

    if process.returncode:
      raise RuntimeError('Couldn\'t clone repository with code {}'.format(process.returncode))

  def __make_index(self):
    # Get filenames index
    self._filenames = set()

    for regex in self.config.file_regexes:
      self._filenames.update(glob.glob(regex, recursive=True))