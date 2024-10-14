from typing import Optional, Literal, Type

from loguru import logger

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
    # <model_name>[@<revision>] or <model_name>
    default_model="Baichuan/Baichuan2_7b_chat_pt@ca161b7",
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
    logger.info(f"framwork is not passed in, use default framework {framework} for task {task}")

  if model is None:
    model = pipeline_wrapper.default_model
    if model is None:
      raise ValueError(f"no default model for {task}")
    if "@" in model:
      model, revision = model.split("@")
    logger.info(f"model is not passed in, use default model {model}")
  
  if backend is None:
    backend = pipeline_wrapper.default_backend
    logger.info(f"backend is not passed in, use default backend {backend}")

  if framework not in PIPELINE_MAPPING[task]:
    raise ValueError(f"{framework} dost not support for {task}")

  if backend not in PIPELINE_MAPPING[task][framework]:
    raise ValueError(f"{backend} does not support for {task}")
  
  pipeline_class = PIPELINE_MAPPING[task][framework][backend]
  pipeline = pipeline_class(task=task,
                            model=model,
                            config=config,
                            tokenizer=tokenizer,
                            feature_extractor=feature_extractor,
                            image_processor=image_processor,
                            revision=revision,
                            framework=framework,
                            backend=backend,
                            **kwargs)
  pipeline_wrapper.set_pipeline(pipeline)
  
  return pipeline_wrapper
  