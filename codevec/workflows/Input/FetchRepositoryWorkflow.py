
from dataclasses import dataclass

import os
import subprocess
import glob
from typing import List, Set

import shutil

from ..Base import AnyWorkflow

class FetchRepositoryWorkflow(AnyWorkflow):

  @dataclass
  class Config:
    repository_url: str
    file_regexes: List[str]

    output_dir: str = 'cache_dir'

  def __init__(self, config: Config):
    super().__init__()

    self.config = config

  def run(self):
    if os.path.exists(self.config.output_dir):
      shutil.rmtree(self.config.output_dir, ignore_errors=True)

    os.makedirs(self.config.output_dir, exist_ok=True)

    wd = os.getcwd()

    os.chdir(self.config.output_dir)

    try:
      self.__clone(self.config)

      os.chdir(wd)

      filenames = self.__make_index(self.config)

      self.update_ctx(dict(processing_filenames = filenames))
    except Exception as exc:
      os.chdir(wd)

      raise exc
    pass

  @staticmethod
  def __clone(config: Config) -> None:
    # Call Git to clone repository
    process = subprocess.Popen(['git', 'clone', '--depth', '1', '--single-branch', config.repository_url],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    process.communicate()

    if process.returncode:
      raise RuntimeError('Couldn\'t clone repository with code {}'.format(process.returncode))

  @staticmethod
  def __make_index(config: Config) -> Set[str]:
    # Get filenames index
    filenames = set()

    for regex in config.file_regexes:
      filenames.update(glob.glob(config.output_dir + '/' + regex, recursive=True))

    return filenames