
from .Node import *
from typing import Generator

import pydot
import os
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

  def __traverse(self, node: Node, validateNode: bool) -> Generator[Node, None, None]:
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
