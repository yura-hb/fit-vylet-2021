from src.Base.Node import Node

def test_node():
  node = Node(None, 'some_str', 'some_name', '', [])

  assert node.is_root == True
  assert node.graph_representation == 'some_name_some_str='
  assert node.is_terminal == True

  new_node = Node(node, 'some_str', 'some_name', '', [])

  assert new_node.is_root == False
  assert new_node.graph_representation == 'some_name_some_str='
  assert new_node.is_terminal == True
