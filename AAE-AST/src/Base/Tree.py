
from torch import rand
from .Node import *
from typing import Generator
from ..Utils.timing import measure

import pydot
import os
import numpy as np
import random

from collections import deque

class Tree:
  """ A tree class, which utilizes a list of nodes and allow to perform specific operations
  """

  def __init__(self, root: Node) -> None:
    """ Initializes the tree with the vector of Nodes

    Args:
        nodes (set[Node]): A set of nodes
    """
    self.root = root

    nodes = list(self.__traverse(root, True))

    self.size = len(nodes)

  def __len__(self) -> int:
    return self.size


  def make_image_graph(self, output_path: str):
    """ Generates image graph of the tree and exports it to the output path.

    Args:
        output_path (path): A path to export image
    """

    assert not os.path.isdir(output_path)

    graph = pydot.Dot(graph_type = 'digraph')

    for node in self.__traverse(self.root, False):
      parent_name = node.graph_representation

      for child in node.children:
        child_name = child.graph_representation

        graph.add_edge(pydot.Edge(parent_name, child_name))

    graph.write_png(output_path)

  def cut(self, node: Node):
    """ Cuts the tree at the specific Node returning a tree object

    Args:
        node (Node): A node to cut.

    Returns:
        A tree object with the root at the node.
    """

    parent = node.parent

    if parent is None:
      return self

    nodes = list(self.__traverse(node, False))
    self.size -= len(nodes)

    node.parent = None
    parent.children.remove(node)

    return Tree(node)

  @measure
  def gen_terminal_paths(self) -> Generator[deque[Node], None, None]:
    """ Generates all paths between terminals in the tree.

      The overall process is done in two steps:

      1. Generates all pathes from root to leaves (BFS Traverse to find leaves)
      2. Selects combinations between terminal nodes and create paths

    Args:
      Node (Terminal): Node shouldn't have any children, a. k. a. it should be terminal

    Returns:
      A generator, which yields a path(deque) between two random leaves in the tree.
    """

    nodes = self.__traverse(self.root, False)
    leaves = list(filter(lambda node: node.is_terminal, nodes))

    def sample(l: list[Node]):
      assert len(l) > 0

      index = random.randint(0, len(l) - 1)
      element = l[index]
      l[index] = l[-1]
      l.pop()

      return element

    while len(leaves) > 1:
      lhs, rhs = sample(leaves), sample(leaves)
      lhs_path, rhs_path = deque(), deque()

      while (lhs != rhs):
        lhs_path.append(lhs)
        lhs = lhs.parent

        rhs_path.appendleft(rhs)
        rhs = rhs.parent

      lhs_path.append(lhs)

      yield lhs_path + rhs_path

  def __traverse(self, node: Node, validateNode: bool = False) -> Generator[Node, None, None]:
    """ Traversers a tree with bfs and returns all closed nods

    Args:
        Node (Node): a root node to start traversing

    Returns:
        set[Node]: A set of all traversed nodes
    """

    nodes = list[Node]()

    q = deque()

    q.append(node)

    while (len(q) != 0):
      node = q.popleft()

      for child in node.children:
        if validateNode and child in nodes:
          raise Exception("A cycle is in the tree for node")

        q.append(child)

      nodes.append(node)

      yield node
