
from ..Base.Node import Node

from enum import Enum

class Namespace(Node):
  """ Namespace is a class, which generalize the idea, that you can declare values inside.
      When we write code, we split it into namespaces, which can be after reused in the code, to perform action.
      So there is no distinction between module, class, func or method in terms of namespace.

      Namespace has a specific traits:
        1. It can inherit from a specific namespace.
        2. It may return value
        3. It may consume some value
  """

  class Kind(Enum):
    CLASS = 0
    FUNC = 1
    MODULE = 2
    FOR = 3
    IF = 4
    WHILE = 5

  def __init__(self,
    parent: Node,
    id: str,
    name: str,
    kind: Kind,
    attributes: list[Node] = [],
    input: list[Node] = [],
    output: list[Node] = [],
    body: list[Node] = []):
    """ Initialize a namespace.

    Args:
        parent (Node): A parent namespace node
        id (str): id of the namespace
        name (str): name of namespace
        input: input to the namespace
        output: output from the namespace
        body: body of the namespace
    """
    self.attributes = attributes
    self.kind = kind
    self.input = input
    self.output = output

    super().__init__(parent, id, name, None, body)


