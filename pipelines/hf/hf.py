import copy
from typing import Optional, List, Union

from loguru import logger

from ..base import BasePipelineWrapper, PTBasePipeline
from .hf_utils import pipeline_registry


class PipelineWrapper(BasePipelineWrapper):
  def __init__(self,
               task: str = None,
               default_framework: str = None,
               default_model: str = None,
               default_backend: str = None,
               **kwargs):
    super().__init__(task, default_framework, default_model, default_backend, **kwargs)


def _get_generated_text(res):
    if isinstance(res, str):
        return res
    elif isinstance(res, dict):
        return res["generated_text"]
    elif isinstance(res, list):
        if len(res) == 1:
            return _get_generated_text(res[0])
        else:
            return [_get_generated_text(r) for r in res]
    else:
        raise ValueError(f"Unsupported result type in _get_generated_text: {type(res)}")


class TextGenerationPipeline(PTBasePipeline):
  def __init__(self,
               task: str,
               model: str,
               revision: str,
               framework: str,
               backend: str,
               **kwargs):
    self.task = "text-generation"
    self.model = model
    self.revision = revision
    self.framework = "pt"
    self.backend = "transformers"
    self.kwargs = copy.deepcopy(kwargs)
    # self.pipeline = None
  
  def init(self):
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
      " model={self.model}, revision={self.revision}).\n"
      "openMind download might take a while, please be patient..."
    )

    try:
      pipeline = pipeline_creator(
        task=self.task,
        model=self.model,
        revision=self.revision,
      )
    except ImportError as e:
      # TODO: add more user-friendly error messages
      raise e
    
    self.pipeline = pipeline

  def _run_pipeline(self, *args, **kwargs):
    # TODO: do we need this?
    try:
      import torch
      # import torch_npu
    except ImportError as e:
      # TODO: add more user-friendly error messages
      raise e

    # autocast causes invalid value (and generates black images) for text-to-image and image-to-image
    no_auto_cast_set = ("text-to-image", "image-to-image")

    if torch.cuda.is_available() and self.task not in no_auto_cast_set:
      with torch.autocast(device_type="cuda"):
        return self.pipeline(*args, **kwargs)
    else:
      return self.pipeline(*args, **kwargs)
    return self.pipeline(*args, **kwargs)
  
  def __call__(
    self,
    inputs: Union[str, List[str]],
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = 1.0,
    repetition_penalty: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    max_time: Optional[float] = None,
    return_full_text: bool = True,
    num_return_sequences: int = 1,
    do_sample: bool = True,
    **kwargs,
  ) -> Union[str, List[str]]:
    res = self._run_pipeline(
      inputs,
      top_k=top_k,
      top_p=top_p,
      temperature=temperature,
      repetition_penalty=repetition_penalty,
      max_new_tokens=max_new_tokens,
      max_time=max_time,
      return_full_text=return_full_text,
      num_return_sequences=num_return_sequences,
      do_sample=do_sample,
      **kwargs,
    )

    return _get_generated_text(res)