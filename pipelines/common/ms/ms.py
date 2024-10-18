import copy
from functools import cached_property
from typing import Callable, List, Union

from loguru import logger

from ...base import MSBasePipeline
from .ms_utils import pipeline_registry


class MSPipeline(MSBasePipeline):
  def __init__(self,
               task: str = None, 
               model: str = None,
               tokenizer: str = None,
               image_processor: str = None,
               audio_processor: str = None,
               backend: str = None,
               **kwargs):
    self.task = task
    self.model = model
    self.tokenizer =tokenizer
    self.image_processor = image_processor
    self.audio_processor = audio_processor
    self.backend = backend
    self.kwargs = copy.deepcopy(kwargs)

    # access pipeline here to trigger download and load
    self.pipeline

  @cached_property
  def pipeline(self) -> Callable:
    try:
      pipeline_creator = pipeline_registry.get(self.task).get(self.framework).get(self.backend)
    except Exception as e:
      # If any error occurs, we issue a warning, but don't exit immediately.
      logger.warning(
        "An error occurred while trying to get pipline creator. Error"
        f" details: {e}"
      )
      pipeline_creator = None
    if pipeline_creator is None:
      raise ValueError(f"Could not find pipeline creator for {self.task}:{self.framework}:{self.backend}")
    
    logger.info(
      f"Creating pipeline for {self.task}(framework={self.framework}, backend={self.backend},"
      f" model={self.model}, revision={self.kwargs.get('revision')}).\n"
      "openMind download might take a while, please be patient..."
    )

    try:
      # TODO: 确认这里要传入什么参数
      pipeline = pipeline_creator(
        task=self.task,
        model=self.model,
        tokenizer=self.tokenizer,
        image_processor=self.image_processor,
        audio_processor=self.audio_processor,
        **self.kwargs,
      )
    except ImportError as e:
      # TODO: add more user-friendly error messages
      raise e
    return pipeline

  def _run_pipeline(self, *args, **kwargs):
    return self.pipeline(*args, **kwargs)


class TextGenerationPipeline(MSPipeline):
  def __init__(self,
               task: str = None, 
               model: str = None,
               tokenizer: str = None,
               image_processor: str = None,
               audio_processor: str = None,
               backend: str = None,
               **kwargs):
    self.task = task
    self.model = model
    self.tokenizer = tokenizer
    self.image_processor = image_processor
    self.audio_processor = audio_processor
    self.framework = "ms"
    self.backend = "mindformers"
    self.kwargs = copy.deepcopy(kwargs)

    # access pipeline here to trigger download and load
    self.pipeline
  
  def __call__(
    self,
    inputs: Union[str, List[str]],
    **kwargs,
  ) -> Union[str, List[str]]:
    res = self._run_pipeline(
      inputs,
      **kwargs,
    )

    return res
