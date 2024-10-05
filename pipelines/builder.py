from typing import Optional, Literal, Type

from .base import BasePipelineWrapper
from .hf import PipelineWrapper, TextGenerationPipeline


# task, BasePipelineWrapper
PIPELINE_WRAPPER_MAPPPING = {}
PIPELINE_MAPPING = {}


def register_pipeline_wrapper(task: str, wrapper: BasePipelineWrapper):
  wrapper.task = task
  PIPELINE_WRAPPER_MAPPPING[task] = wrapper


def register_pipeline(task: str, framework: str, backend: str, pipeline: Type, **kwargs):
  if task not in PIPELINE_MAPPING:
    PIPELINE_MAPPING[task] = {}
    if framework not in PIPELINE_MAPPING[task]:
      PIPELINE_MAPPING[task][framework] = {}
    PIPELINE_MAPPING[task][framework][backend] = pipeline


register_pipeline_wrapper(
  "text-generation",
  PipelineWrapper(
    default_model="/home/lynn/github/qwen2.5-0.5b-instruct",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline(
  "text-generation",
  "pt",
  "transformers",
  TextGenerationPipeline
)

def get_pipeline_wrapper(
  task: Optional[str] = None,
  model: Optional[str] = None,
  config: Optional[str] = None,
  tokenizer: Optional[str] = None,
  feature_extractor: Optional[str] = None,
  image_processor: Optional[str] = None,
  revision: Optional[str] = None,
  framework: Optional[Literal["pt", "ms"]] = None,
  backend: Optional[str] = None,
  **kwargs,
):
  if task is None and model is not None:
    raise ValueError("openmind currently does not support creating a pipeline object by only passing in the model")
  
  pipeline_wrapper = PIPELINE_WRAPPER_MAPPPING.get(task)

  if framework is None:
    framework = pipeline_wrapper.default_framework
    if framework is None:
      raise ValueError(f"no default framework for {task}")

  if model is None:
    model = pipeline_wrapper.default_model
    if model is None:
      raise ValueError(f"no default model for {task}")
  
  if backend is None:
    backend = pipeline_wrapper.backend

  if framework not in PIPELINE_MAPPING[task]:
    raise ValueError(f"{framework} dost not support for {task}")

  if backend not in PIPELINE_MAPPING[task][framework]:
    raise ValueError(f"{backend} does not support for {task}")
  
  pipeline_class = PIPELINE_MAPPING[task][framework][backend]
  pipeline = pipeline_class(task=task,
                            model=model,
                            revision=revision,
                            framework=framework,
                            backend=backend,
                            **kwargs)
  
  pipeline.init()
  pipeline_wrapper.set_pipeline(pipeline)
  
  return pipeline_wrapper
  