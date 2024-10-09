from loguru import logger

from .utils import Registry


task_cls_registry = Registry()


class PipelineBuilder:
  # please override this in your derived class
  task: str = "undefined"
  default_model_id: str = "undefined"
  framework: str = "undefined"
  backend: str = "undefined"

  def __init_subclass__(cls, **kwargs) -> None:
    # 注意：这里可能会有冲突
    task_cls_registry.register(
      cls.task,
      {
        cls.framework: {
          cls.backend: cls,
        },
      }
    )
  
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
  def create(cls, task, model_str, revision, framework, backend, **kwargs):
    raise NotImplementedError
