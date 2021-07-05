
from ..Base.Node import Node

class Constant(Node):

  def __init__(self, parent, id: str, type: str, value: str) -> None:
    name = 'const' if type == None else 'const_' + type

    super().__init__(parent, id, name, value, [])
