from typing import Optional, Literal, Type, Dict, Any

from loguru import logger


from .base import pipeline_wrapper_registry, pipeline_registry
from .utils import get_task_from_readme

# pipelinewrapper & pipeline registrations are done by pre-importing the module
from .common import *  # noqa: F403


def get_pipeline_wrapper(
  task: Optional[str] = None,
  model: Optional[str] = None,
  config: Optional[str] = None,
  tokenizer: Optional[str] = None,
  feature_extractor: Optional[str] = None,
  image_processor: Optional[str] = None,
  framework: Optional[Literal["pt", "ms"]] = None,
  backend: Optional[str] = None,
  model_kwargs: Dict[str, Any] = None,
  **kwargs,
):
  if task is None and model is None:
    raise RuntimeError(
      "Impossible to instantiate a pipeline without either a task or a model being specified. "
      "Please provide a task class or a model"
      )
  
  if model is None and tokenizer is not None:
    raise RuntimeError(
      "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided tokenizer"
      " may not be compatible with the default model. Please provide a PreTrainedModel class or a"
      " path/identifier to a pretrained model when providing tokenizer."
    )
  
  if model is None and feature_extractor  is not None:
    raise RuntimeError(
      "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the provided"
      " feature_extractor may not be compatible with the default model. Please provide a PreTrainedModel class"
      " or a path/identifier to a pretrained model when providing feature_extractor."
    )
  
  if model is None and image_processor is not None:
    raise RuntimeError(
      "Impossible to instantiate a pipeline with image_processor specified but not the model as the provided"
      " feature_extractor may not be compatible with the default model. Please provide a PreTrainedModel class"
      " or a path/identifier to a pretrained model when providing image_processor."
    )
  
  if task is None and model is not None:
      if isinstance(model, str):
        task = get_task_from_readme(model)
      else:
        raise RuntimeError(
          "task must be provided when the type of model is PreTrained model"
        )
  
  pipe_wrapper_cls = pipeline_wrapper_registry.get(task)

  if framework is None:
    framework = pipe_wrapper_cls.framework
    if framework is None:
      raise RuntimeError(
        "Framework is not specified and the pipeline wrapper for {task} does not specify a default framework"
      )
    logger.info(f"Framework is not specified, use default framework {framework} for task {task}")

  if model is None:
    model = pipe_wrapper_cls.model_id
    if model is None:
      raise ValueError(f"no default model for {task}")
    if "@" in model:
      model, revision = model.split("@")
      kwargs["revision"] = revision
    logger.info(f"Model is not specified, use default model {model}")
  
  if backend is None:
    backend = pipe_wrapper_cls.backend
    logger.info(f"backend is not passed in, use default backend {backend}")

  if framework not in pipeline_registry.get(task):
    raise ValueError(f"{framework} dost not support for {task}")

  if backend not in pipeline_registry.get(task).get(framework):
    raise ValueError(f"{backend} does not support for {task}")
  
  pipeline_class =  pipeline_registry.get(task).get(framework).get(backend)
  pipeline = pipeline_class(task=task,
                            model=model,
                            config=config,
                            tokenizer=tokenizer,
                            feature_extractor=feature_extractor,
                            image_processor=image_processor,
                            framework=framework,
                            backend=backend,
                            model_kwargs=model_kwargs,
                            **kwargs)

  return pipe_wrapper_cls(pipeline)
