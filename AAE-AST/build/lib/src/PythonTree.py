
from .Tree import *

class PythonTree(Tree):

  #
  # Need to implement:
  #   1. Init with JSON
  #   2. Variable obfuscation
  #   3. Path generation -> May be in tree
  #   4. subtree cut technique, using types
  #

  _type = 'type'
  _children = 'children'
  _value = '_value'


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

    nodes = []

    def dfs(parent: Node, item_json: dict):
      if isinstance(item_json, dict) and self._children in item_json.keys():

        for child in item_json[self._children]:
          child_json = json[child]
          node = Node(parent, str(child), child_json[self._type], child_json[self._value], [])
          node = dfs(node, child_json)

          parent.children.append(node)

      nodes.append(parent)

      return parent

    entry = json[0]

    assert self._type in entry.keys()

    root = Node(None, str(0), entry[self._type], entry[self._value], [])

    dfs(root, entry)

    super().__init__(root, nodes)
