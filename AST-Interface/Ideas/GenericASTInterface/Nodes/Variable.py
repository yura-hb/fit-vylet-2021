
from enum import Enum
from ..Base.Node import Node

class Variable(Node):
  """ Variable is a class, which can be loaded, stored or deleted. That's all

  Args:
      Node ([type]): [description]
  """

  class Operation(Enum):
    LOAD = 0
    STORE = 1
    DELETE = 2

  def __init__(self, parent, id: str, operation: Operation, name: str, value: str, children) -> None:
    self.operation = operation

    super().__init__(parent, id, name, value, children)
