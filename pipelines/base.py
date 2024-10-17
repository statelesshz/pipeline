from typing import Callable, Optional

from .utils import Registry


pipeline_wrapper_registry = Registry()


class BasePipelineWrapper:
   task: Optional[str] = None
   framework: Optional[str] = None
   backend: Optional[str] = None
   # model_id spec should be in the form of <model_id>[@<revision>], or
   # <model_id>
   model_id: Optional[str] = None

   def __init__(self, pipeline: Callable, **kwargs):
      self.pipeline = pipeline


   def __init_subclass__(cls, **kwargs) -> None:
      super().__init_subclass__(**kwargs)
      if cls.task is None:
         raise ValueError("You made a programming error: the task is None.")
      if cls.framework is None:
         raise ValueError("You made a programming error: the framework is None.")
      if cls.backend is None:
         raise ValueError("You made a programming error: the backend is None")
      pipeline_wrapper_registry.register(
         cls.task, cls
      )

   def __call__(self, *args, **kwargs):
      return self.pipeline(*args, **kwargs)


class BasePipeline:
   def __init__(self, task, **kwargs):
      self.task = task

class PTBasePipeline(BasePipeline):
   def __init__(self, task, **kwargs):
      super().__init__(task, **kwargs)
      self.framework = "pt"


class MSBasePipeline(BasePipeline):
   def __init__(self, task, **kwargs):
      super().__init__(task, **kwargs)
      self.framework = "ms"
