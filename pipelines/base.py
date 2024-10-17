
from abc import ABC


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

class PTBasePipeline(BasePipeline):
   def __init__(self,
                task: str,
                framework: str = None,
                **kwargs):
      super().__init__(task, **kwargs)
      self.framework = "pt"


class MSBasePipeline(BasePipeline):
   def __init__(self,
                task: str,
                **kwargs):
      super().__init__(task, **kwargs)
      self.framework = "ms"
