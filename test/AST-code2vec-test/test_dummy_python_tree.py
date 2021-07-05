
from src.Base.Tree import Tree
from src.Base.Node import Node
from src.Derived.DummyPythonTree import DummyPythonTree

from typing import Any

import os
import json

from yaml import load

class TestDummyPythonTree:

  def test_tree_parsing(self):
    js = self.load_json()

    try:
      os.rmdir('images_org/')
    except:
      pass

    try:
      os.mkdir('images_org/')
    except Exception as e:
      print('Exception during folder creation: ', e)

    for i in range(0, len(js)):
      tree = DummyPythonTree.from_json(js[i])
      tree.make_image_graph('images_org/image_{}.png'.format(i))

      for path in tree.gen_terminal_paths():
        print(list(map(lambda x: x.__str__(), path)))

  def test_tree_split_parsing(self):
    js = self.load_json()

    iterator = 0

    try:
      os.rmdir('images_spl/')
    except:
      pass

    try:
      os.mkdir('images_spl')
    except Exception as e:
      print('Exception during folder creation: ', e)

    for i in range(0, len(js)):
      tree = DummyPythonTree.from_json(js[i])

      for new_tree in tree.split():
        new_tree.make_image_graph('images_spl/image_{}.png'.format(iterator))
        iterator += 1

      tree.make_image_graph('images_spl/image_{}.png'.format(iterator))
      iterator += 1

  def load_json(self) -> list[Any]:
    assert os.path.exists('test.json'), "No json file found"

    js = []

    with open('test.json') as f:
      for line in f:
        js.append(json.loads(line))

    return js
