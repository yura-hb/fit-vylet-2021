
from numpy import append
import re

from ..Base.Node import Node
from ..Nodes.Namespace import Namespace
from ..Nodes.Import import Import
from ..Nodes.Constant import Constant
from ..Nodes.Dummy import Dummy

from ..Base.Tree import Tree

class PythonTree(Tree):

  """ A class, which maps input from python100k ast dataset, where each field is in format:
        { 'type': 'any', 'value': 'any', 'children': [any number]  }
  """

  _type = 'type'
  _children = 'children'
  _value = 'value'


  #
  # Import, ImportFrom is always followed with the aliases, so it can be reduced in one node as a part of research.
  # Call is always has as child: NameLoad to load name, attributes, keywords.
  # Function is always followed with arguments, body and decorator_list.
  # Class is always followed with bases, body and decorator_list
  # For is always followed with store, load, body, orelse
  # While is always followed with load, body, orelse
  #
  ast_map = {
    'Module': 'Module',
    'Expr': 'Expr',

    'Str': 'Constant',
    'Num': 'Constant',

    # Base objects
    'Dict': 'Dict',
    'Set': 'Set',

    # Definitions
    'FuncDef': 'FuncDef',
    'ClassDef': 'ClassDef',

    # Import
    'Import': 'Import',
    'ImportFrom': 'ImportFrom',

    # Class + Store/Load/Delete
    'NameStore': 'Store',
    'SubscriptStore': 'Store',
    'TupleStore': 'Store',
    'AttributeStore': 'Store',
    'ListStore': 'Store',
    'NameLoad': 'Load',
    'AttributeLoad': 'Load',
    'ListLoad': 'Load',
    'TupleLoad': 'Load',
    'SubscriptLoad': 'Load',
    'NameDelete': 'Delete',
    'Delete': 'Delete',
    'SubscriptDel': 'Delete',
    'NameDel': 'Delete',
    'AttributeDel': 'Delete',

    # Default keywords
    'Return': 'Return',
    'Pass': 'Pass',
    'For': 'For',
    'If': 'If',
    'IfExpr': 'IfExpr',
    'While': 'While',
    'With': 'With',
    'Continue': 'Continue',
    'Raise': 'Raise',
    'Break': 'Break',
    'TryExcept': 'TryExcept',
    'TryFinally': 'TryFinally',
    'ExceptHandler': 'ExceptHandler',
    'Lambda': 'Lambda',
    'Assert': 'Assert',
    'Yield': 'Yield',
    'Global': 'Global',
    'Exec': 'Exec',
    'Repr': 'Repr',

    # Comprehensions
    'SetComp': 'SetComp',
    'ListComp': 'ListComp',
    'GeneratorExp': 'GeneratorExp',
    'DictComp': 'DictComp',
    'comprehension': 'comprehension',


    # Function/Object call
    'Call': 'Call',
    'Print': 'Print', # TODO: Should be changed on call

    # Binary Operators
    'Assign': 'Assign',

    ## op =
    'AugAssignAdd': 'AugAssignAdd',
    'AugAssignSub': 'AugAssignSub',
    'AugAssignMult': 'AugAssignMult',
    'AugAssignDiv': 'AugAssignDiv',
    'AugAssignFloorDiv': 'AugAssignFloorDiv',
    'AugAssignMod': 'AugAssignMod',
    'AugAssignPow': 'AugAssignPow',
    'AugAssignLShift': 'AugAssignLShift',
    'AugAssignRShift': 'AugAssignRShift',

    'AugAssignBitAnd': 'AugAssignBitAnd',
    'AugAssignBitOr': 'AugAssignBitOr',
    'AugAssignBitXor': 'AugAssignBitXor',

    'AugAssignMatMult': 'AugAssignMatMult',

    # op operand
    'UnaryOpInvert': 'UnaryOpInvert',
    'UnaryOpNot': 'UnaryOpNot',
    'UnaryOpUSub': 'UnaryOpUSub',
    'UnaryOpUAdd': 'UnaryOpUAdd',

    # op operand op
    'BinOpAdd': 'BinOpAdd',
    'BinOpSub': 'BinOpSub',
    'BinOpMult': 'BinOpMult',
    'BinOpDiv': 'BinOpDiv',
    'BinOpFloorDiv': 'BinOpFloorDiv',
    'BinOpMod': 'BinOpMod',
    'BinOpPow': 'BinOpPow',
    'BinOpLShift': 'BinOpLShift',
    'BinOpRShift': 'BinOpRShift',
    'BinOpBitAnd': 'BinOpBitAnd',
    'BinOpBitOr': 'BinOpBitOr',
    'BinOpBitXor': 'BinOpBitXor',

    'BoolOpAnd': 'BoolOpAnd',
    'BoolOpOr': 'BoolOpOr',

    'CompareEq': 'CompareEq',
    'CompareNotEq': 'CompareNotEq',
    'CompareLt': 'CompareLt',
    'CompareLtE': 'CompareLtE',
    'CompareGt': 'CompareGt',
    'CompareGtE': 'CompareGtE',
    'CompareIs': 'CompareIs',
    'CompareIsNot': 'CompareIsNot',
    'CompareIn': 'CompareIn',
    'CompareNotIn': 'CompareNotIn',

    # Slices
    'Slice': 'Slice',
    'ExtSlice': 'ExtSlice',
    'Index': 'Index',
    'Ellipsis': 'Ellipsis',

    # TODO: - Think what to do with it.
    'attr': 'attribute',
    'args': 'args',
    'arguments': 'arguments',
    'NameParam': 'NameParam',
    'defaults': 'defaults',
    'body': 'body',
    'decorator_list': 'decorator_list',
    'bases': 'bases',
    'orelse': 'orelse',
    'handlers': 'handlers',
    'type': 'type',
    'keyword': 'keyword',
    'identifier': 'identifier',
    'vararg': 'vararg',
    'kwarg': 'kwarg',
    'finalbody': 'finalbody',
    'name': 'name'
  }

  def __init__(self, json):
    """ Creates a tree with json.

    Args:
        json (dict): A Python AST preprocessed json with the next fields for each node:
          1. type -> type of the node
          2. value -> value, which node stores
          3. children -> a id of nodes in the json
    """

    assert not len(json) == 0

    root = PythonTree.__make_node('0', json[0], json)[0]

    super().__init__(root)

  #
  # Mapping. From python ast data to abstract ast representation.
  #

  @staticmethod
  def __make_node(id: str, item_json: dict, json: list) -> list[Node]:
    assert PythonTree._type in item_json.keys()

    void = lambda x: x

    map = {
      r'Module': PythonTree.__module_node,
      r'(FuncDef)|(FunctionDef)': PythonTree.__func_node,
      r'ClassDef': PythonTree.__class_node,
      r'(Str)|(Num)': PythonTree.__constant_node,
      r'(Import)|(ImportFrom)': PythonTree.__import_node,
      r'(Call)': PythonTree.__import_node,
      #r'.*((Store)|(Load)|(Delete)|(Del))$': void,
      #r'(Return)|(Pass)|(Continue)|(Raise)|(Break)': void
    }

    type = item_json[PythonTree._type]

    for regex, func in map.items():
      if re.match(regex, type):
        return func(id, item_json, json)

    node = Dummy(None, id + '_' + repr(item_json), [])

    node.children += PythonTree.__traverse_children(id, item_json, json)

    return [node]

  @staticmethod
  def __module_node(id: str, item_json: dict, json: list) -> list[Node]:
    attributes = []
    body = []

    # In case, if module has children we perform calls
    if PythonTree._children in item_json.keys():
      for child in item_json[PythonTree._children]:
        for node in PythonTree.__make_node(str(child), json[child], json):
          if node is Import:
            attributes.append(node)
          else:
            body.append(node)


    result = Namespace(None, id, 'module', Namespace.Kind.MODULE, attributes, [], [], body)

    for node in attributes + body:
      node.parent = result

    return [result]

  @staticmethod
  def __func_node(id: str, item_json: dict, json: list) -> list[Node]:
    name = item_json[PythonTree._value]

    attributes = []
    input = []
    body = []

    if PythonTree._children in item_json.keys():
      for child in item_json[PythonTree._children]:
        node = json[child]

        if node[PythonTree._type] == 'arguments':
          # TODO: - Implement
          continue

        if node[PythonTree._type] == 'decorator_list':
          # TODO: - Implement
          continue

        if node[PythonTree._type] == 'body':
          body += PythonTree.__traverse_children(child, node, json)
          continue

        print('Unexpected node in function {}'.format(node))

    result = Namespace(None, id, name, Namespace.Kind.FUNC, attributes, input, [], body)

    for node in attributes + input + body:
      node.parent = result

    return [result]

  @staticmethod
  def __class_node(id: str, item_json: dict, json: list) -> list[Node]:
    name = item_json[PythonTree._value]

    attributes = []
    body = []

    # In case, if module has children we perform calls
    if PythonTree._children in item_json.keys():
      for child in item_json[PythonTree._children]:
        node = json[child]

        if node[PythonTree._type] == 'bases':
          # TODO: - Implement
          continue

        if node[PythonTree._type] == 'decorator_list':
          # TODO: - Implement
          continue

        if node[PythonTree._type] == 'body':
          body += PythonTree.__traverse_children(child, node, json)
          continue

        print('Unexpected node in class {} {}'.format(item_json, node))

    result = Namespace(None, id, name, Namespace.Kind.CLASS, attributes, [], [], body)

    for node in attributes + body:
      node.parent = result

    return [result]

  @staticmethod
  def __constant_node(id: str, item_json: dict, json: list):
    return [Constant(None, id, item_json[PythonTree._type].lower(), item_json.get(PythonTree._value))]

  @staticmethod
  def __import_node(id: str, item_json: dict, json: list):
    if item_json[PythonTree._type] == 'Import':
      assert len(item_json[PythonTree._children]) == 1
      import_alias = json[item_json[PythonTree._children][0]]
      assert import_alias[PythonTree._type] == 'alias'

      return [Import(None, id, alias = import_alias[PythonTree._value])]

    if item_json[PythonTree._type] == 'ImportFrom':
      nodes = []

      for child in item_json[PythonTree._children]:
        import_alias = json[child]

        assert import_alias[PythonTree._type] == 'alias'

        value = item_json[PythonTree._value]

        node = Import(None, str(child), fromModule = value, alias = import_alias[PythonTree._value])

        nodes.append(node)

      return nodes

    return [Dummy(None, id + '_' + repr(item_json), [])]

  @staticmethod
  def __call(id: str, item_json: dict, json: list):


  @staticmethod
  def __traverse_children(id: str, item_json: dict, json: list):
    nodes = []

    if PythonTree._children in item_json.keys():
      for node_child in item_json[PythonTree._children]:
        nodes += PythonTree.__make_node(str(node_child), json[node_child], json)

    return nodes
