
from abc import ABC
from typing import List, Optional, Sequence


class BasePipelineWrapper(ABC):
   def __init__(self,
               task: str,
               default_framework: str = None,
               default_model: str = None,
               default_backend: str = None,
               **kwargs):
      self.task = task
      # <model_id>[@<revision>]
      self.default_model = default_model
      self.default_framework = default_framework
      self.default_backend = default_backend

   def set_pipeline(self, pipeline):
      self.pipeline = pipeline
   
   def __call__(self, *args, **kwargs):
      return self.pipeline(*args, **kwargs)


class BasePipeline(ABC):
   def __init__(self,
                task: str,
                **kwargs):
      self.task = task

      self.framework = None
      self.model = None
      self.revision = None
      self.framework = None
      self.backend = None

   def init(self, model_str, *args, **kwargs):
      raise NotImplementedError
   

class PTBasePipeline(BasePipeline):
   def __init__(self,
                task: str,
                **kwargs):
      super().__init__(task, **kwargs)
      self.framework = "pt"
