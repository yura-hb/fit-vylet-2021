
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
    root = self.parent

    while root.parent:
      root = root.parent

    current = root

    ctx = self.global_ctx

    while current:
      print('Start running workflow with name {}'.format(type(current)))
      current.run(ctx)
      print('End running workflow with name {}'.format(type(current)))

      ctx.update(current.global_ctx)

      current = current.next

    return ctx

  def run(self, ctx):
    assert False, "Must be implemented"