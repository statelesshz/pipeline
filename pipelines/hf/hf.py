import copy
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from ..base import PipelineBuilder, task_cls_registry
from .hf_utils import pipeline_registry


HF_DEFINED_TASKS = [
  # diffusers
  "text-to-image",
  # transformers
  "text-classification",
  "text-generation",
]


class HFPipelineBuilder(PipelineBuilder):
  def __init_subclass__(cls, **kwargs) -> None:
    super().__init_subclass__(**kwargs)
    if cls.task not in HF_DEFINED_TASKS:
      raise ValueError(
        f"You made a programming error: the task {cls.task} is not a"
        " supported task defined in HuggingFace. If you believe this is an"
        " error, please file an issue."
      )
    task_cls_registry.register(
      cls.task,
      {
        cls.framework: {
          cls.backend: cls,
        }
      }
    )

  def __init__(self, task, model_str: str, revision, framework, backend, **kwargs):
    self.task = task
    self.model = model_str
    self.revision = revision
    self.framework = framework
    self.backend = backend

    self.kwargs = copy.deepcopy(kwargs)
    # access pipeline here to trigger download and load
    self.pipeline

  @cached_property
  def pipeline(self):
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
    return pipeline
    
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
  
  def __call__(self, *args, **kwargs):
    return self._run_pipeline(*args, **kwargs)

  @classmethod
  def create(cls, task, model_str, revision, framework, backend, **kwargs):
    # task, framework, backend = cls._parse_model_str(task, model_str, framework, backend, **kwargs)
    try:
      task_cls = task_cls_registry.get(task).get(framework).get(backend)
    except Exception as e:
      logger.warning(
        "An error occurred while trying to get pipeline builder. Error"
        f" details: {e}"
      )
      task_cls = None
    if task_cls is None:
      raise ValueError(
        f"openmind current does not support the secified task: {task}. If you would"
        "like us to support this task. please let us know by opening an issue at"
        " https://xxx, and kindly include the specific odel that you are trying to run"
        " for debugging purpose: {model_str}"
      )
    return task_cls(task, model_str, revision, framework, backend, **kwargs)


class HFTextClassificationBuilder(HFPipelineBuilder):
  task = "text-classification"
  # <model_name>[@<revision>]
  default_model_id = "/home/lynn/github/distilbert-base-uncsed"
  framework = "pt"
  backend = "transformers"

  def __call__(
    self,
    inputs: Union[str, List[str]],
    **kwargs,
  ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    return self._run_pipeline(inputs, **kwargs)
  

class HFTextToImageBuilder(HFPipelineBuilder):
  task = "text-to-image"
  framework = "pt"
  backend = "diffusers"

  def __call__(
    self,
    prompt: Union[str, List[str]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    seed: Optional[Union[int, List[int]]] = None,
    **kwargs,
  ):
    import torch
    if torch.cuda.is_available():
      self._device = "cuda"
    else:
      self._device = "npu"
    
    if seed is not None:
      if not isinstance(seed, list):
          seed = [seed]
      generator = [
          torch.Generator(device=self._device).manual_seed(s) for s in seed
      ]
    else:
      generator = None

    return self._run_pipeline(
      prompt,
      height=height,
      width=width,
      num_inference_steps=num_inference_steps,
      guidance_scale=guidance_scale,
      negative_prompt=negative_prompt,
      generator=generator,
      **kwargs,
    ).images[0]
