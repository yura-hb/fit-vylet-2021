from ..Base.Node import Node

class Call(Node):

  def __init__(self, parent, id: str, value: str, children) -> None:
      super().__init__(parent, id, 'call', value, children)
