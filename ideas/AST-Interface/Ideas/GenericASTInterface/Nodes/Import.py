
from ..Base.Node import Node
from .Namespace import Namespace

from enum import Enum

class Import(Node):
  """ Import is a dependency, which adds info to the Namespace.
  """

  def __init__(self, parent: Namespace, id: str, fromModule: str = '', alias: str = '', name: str = ''):
    self.fromModule = fromModule
    self.alias = alias
    self.asname = name

    super().__init__(parent, id, 'import', None, [])
