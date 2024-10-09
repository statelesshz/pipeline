from loguru import logger

from .utils import Registry


task_cls_registry = Registry()


class PipelineWrapper:
  # please override this in your derived class
  task: str = "undefined"
  model_name_or_path: str = "undefined"
  framework: str = "undefined"
  backend: str = "undefined"
  
  @classmethod
  def supported_tasks(cls):
    """
    Returns the set of supported tasks.
    """
    return task_cls_registry.keys()

  def __call__(self, *args, **kwargs):
    return self.pipeline(*args, **kwargs)

  def pipeline(self):
    raise NotImplementedError
  
  @classmethod
  def build(cls, task, model_str, revision, framework, backend, **kwargs):
    raise NotImplementedError
