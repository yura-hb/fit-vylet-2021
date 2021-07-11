from gettext import find
from setuptools import setup, find_packages

setup(
  name = 'codevec',
  version = '0.0.1',
  license = 'MIT',
  package_dir = { "": "codevec" },
  packages = find_packages("codevec")
)
