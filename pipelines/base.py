from functools import cached_property

from loguru import logger

from .utils import Registry


pipeline_registry = Registry()

task_cls_registry = Registry()


class PipelineBuilder:
  task: str = "undefined (please override this in your derived class)"
  default_model_id: str = "undefined (please override this in your derived class)"
  framework: str = "undefined (please override this in your derived class)"
  backend: str = "undefined (please override this in your derived class)"

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
  
  @classmethod
  def _parse_model_str(cls, model_str):
    # model_str
    # hf:<model_name>[@<revision>]
    # hf:<task_name>:<model_name>[@<revision>]
    task = cls.task
    model_id = cls.default_model_id
    revision = None
    backend = cls.backend
    framework = cls.framework
    return task, model_id, revision, framework, backend
  
  def __init__(self, model_str: str):
    task, model_id, revision, _, _ = self._parse_model_str(model_str)
    self.model = model_id
    self.task = task
    self.revision = revision
    # access pipeline here to trigger download and load
    self.pipeline

  def __call__(self, *args, **kwargs):
    return self.pipeline(*args, **kwargs)

  @cached_property
  def pipeline(self):
    raise NotImplementedError
  
  @classmethod
  def create_from_model_str(cls, model_str):
    raise NotImplementedError
