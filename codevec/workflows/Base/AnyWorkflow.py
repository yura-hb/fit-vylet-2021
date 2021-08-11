
from typing import Dict, Any

class AnyWorkflow:

  def __init__(self):
    self.global_ctx = {}
    self.next = None
    self.parent = None

  def pipe(self, workflow: 'AnyWorkflow') -> 'AnyWorkflow':
    assert self.next is None, "Workflows must be connected"

    self.next = workflow
    workflow.parent = self

    return workflow

  def execute(self):
    root = self.parent or self

    while root.parent:
      root = root.parent

    current = root

    ctx = self.global_ctx

    while current:
      print('Start running workflow with name {}'.format(type(current)))
      current.run()
      print('End running workflow with name {}'.format(type(current)))

      ctx.update(current.global_ctx)

      current = current.next

    return ctx

  def get_from_ctx(self, key) -> Any:
    return self.global_ctx[key]

  def update_ctx(self, dict: Dict):
    print(self.global_ctx)

    self.global_ctx.update(dict)

    print(self.global_ctx)

  def run(self):
    assert False, "Must be implemented"