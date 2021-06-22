
from ..Base.Node import Node

class Dummy(Node):

  def __init__(self, parent: Node, value: str, children: list[Node]) -> None:
      super().__init__(parent, '', 'Dummy', value, children)
