
import pytest
import numpy as np
from numpy.testing import assert_equal
from collections import deque
from PIL import Image

from src.Base.Tree import Tree
from src.Base.Node import Node

def test_tree():
  tree = make_tree()

  with pytest.raises(Exception, match=r'cycl'):
    tree = make_cyclic_tree()

def test_export():
  tree = make_tree()

  tree.make_image_graph('image.png')

  assert_equal(read_image('image.png'), read_image('source.png'))

def test_paths():
  tree = make_tree()

  paths = list(tree.gen_terminal_paths())

  assert len(paths) == 1, "The tree with two leaves must return 1 path"
  assert deque([tree.root.children[0], tree.root, tree.root.children[1]]) in paths

def test_cut():
  tree = make_tree()

  tree_nodes = len(tree)

  node = tree.root.children[1]
  parent = node.parent

  children_count = len(parent.children)

  new_tree = tree.cut(node)

  assert new_tree.root == node
  assert node.parent == None
  assert len(parent.children) == children_count - 1
  assert len(tree) == tree_nodes - 1

def make_tree() -> Tree:
  left_child = Node(None, 'left', 'str', 'str', [])
  right_child = Node(None, 'right', 'str', 'str', [])

  root = Node(None, 'root', 'str', 'str', [])

  root.children = [left_child, right_child]
  left_child.parent = root
  right_child.parent = root

  return Tree(root)


def make_cyclic_tree() -> Tree:
  left_child = Node(None, 'left', 'str', 'str', [])
  right_child = Node(None, 'right', 'str', 'str', [])

  root = Node(None, 'root', 'str', 'str', [])

  root.children = [left_child, right_child]
  left_child.parent = root
  right_child.parent = root

  left_child.children = [right_child]
  right_child.children = [left_child]

  return Tree(root)

def read_image(file_name: str) -> np.array:
  return np.asarray(Image.open(file_name), dtype=np.uint8)
