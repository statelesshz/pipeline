from typing import Callable, Optional, List

from .utils import Registry


pipeline_wrapper_registry = Registry()
pipeline_registry = Registry()


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
   task: Optional[str] = None
   framework: Optional[str] = None
   backend: Optional[str] = None
   requirement_dependency: Optional[List[str]] = None

   def __init_subclass__(cls, **kwargs) -> None:
      super.__init_subclass__(**kwargs)
      if cls.task is None:
         raise ValueError("You made a programming error: the task is None.")
      if cls.framework is None:
         raise ValueError("You made a programming error: the framework is None.")
      if cls.backend is None:
         raise ValueError("You made a programming error: the backend is None")
      if cls.task not in pipeline_registry.keys():
         pipeline_registry.register(
            cls.task, {
               cls.framework: {
                  cls.backend: cls,
               },
            }
         )
      else:
         if cls.framework not in pipeline_registry.get(cls.task):
            pipeline_registry.get(cls.task)[cls.framework] = {}
         pipeline_registry.get(cls.task)[cls.framework][cls.backend] = cls


class PTBasePipeline(BasePipeline):
   framework: str = "pt"
   # please override this in your derived class
   task: str = "undefined"
   backend: str = "undefined"


class MSBasePipeline(BasePipeline):
   framework: str = "ms"
   # please override this in your derived class
   task: str = "undefined"
   backend: str = "undefined"
