
class Node:

  def __init__(self, parent, id: str, name: str, value: str, children) -> None:
    """Initializes the noe type

    Args:
        parent (Node): node parent
        name (str): node name
        value (str): node value
        children (list[Node]): node children
    """

    self.parent = parent
    self.id = id
    self.name = name
    self.value = value
    self.children = children

  def __str__(self):
    return self.graph_representation

  @property
  def is_root(self):
    return self.parent == None

  @property
  def is_terminal(self):
    """ Validates, if the Node is a terminal

    Returns:
        bool
    """
    return len(self.children) == 0

  @property
  def graph_representation(self):
    if self.value == None:
      return self.name + "_" + self.id

    return self.name + "_" + self.id + "=" + self.value

  def obfuscate(self):
    pass

  def obfuscate_name(self):
    pass

  def obfuscate_value(self):
    pass