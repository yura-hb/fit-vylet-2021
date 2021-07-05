from src.Base.Tree import Tree
from src.Base.Node import Node
from src.Derived.PythonTree import PythonTree

from typing import Any

import os
import json

from yaml import load

def test_tree_parsing():
  js = load_json()

  for i in range(0, len(js)):
    tree = PythonTree(js[i])
    tree.make_image_graph('image_{}.png'.format(i))

def load_json() -> list[Any]:
  assert os.path.exists('test.json'), "No json file found"

  js = []

  with open('test.json') as f:
    for line in f:
      js.append(json.loads(line))

  return js
