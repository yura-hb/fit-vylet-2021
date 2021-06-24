
from ..Base.Tree import Tree
from ..Base.Node import Node

from ..Utils.timing import measure

class DummyPythonTree(Tree):

  _type, _value, _children = 'type', '_value', 'children'

  """ A dummy implementation of the Tree, which want to achieve the next goals:

  1. Split tree on classes, functions, methods
  2. Load tree from json
  """

  @measure
  def __init__(self, root: Node):
    super().__init__(root)

  @staticmethod
  def from_json(json: list):
    """ Creates a tree with json.

    Args:
        json (dict): A Python AST preprocessed json with the next fields for each node:
          1. type -> type of the node
          2. value -> value, which node stores
          3. children -> a id of nodes in the json
    """

    assert len(json) > 0

    def dfs(node: Node, id: str, item_json, json):
      child_node = Node(node, id, item_json[DummyPythonTree._type], item_json.get(DummyPythonTree._value), [])

      if DummyPythonTree._children in item_json.keys():
        for child_id in item_json[DummyPythonTree._children]:
          child_node.children.append(dfs(child_node, str(child_id), json[child_id], json))

      return child_node

    root = dfs(None, '0', json[0], json)

    return DummyPythonTree(root)

  def split(self):
    """ A simple method, which will traverse input tree and try to split it.

    The idea behind this method is to simplify the ast and potentially reduce the amount of paths between unneccesary blocks of
    code.

    The method ensures, that you will have at least one Class or Func in the contextю

    Return:
        A generator, which will yield a new tree
    """

    did_encounter_func = did_encounter_class = False

    for node in self.traverse(self.root):

      if node.name == 'ClassDef':
        if not did_encounter_class:
          did_encounter_class = True
          continue

        yield DummyPythonTree(self.cut(node, True))
        continue

      if node.name == 'FunctionDef':
        if not did_encounter_func:
          did_encounter_func = True
          continue

        yield DummyPythonTree(self.cut(node, True))
        continue
