from typing import Optional, Literal, Type, Dict, Any

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
    # TODO: support instantiate a pipeline with model specified only
    raise RuntimeError(
      "Impossible to instantiate a pipelie with model specidifed but not the task."
    )
  
  pipe_wrapper = PIPELINE_WRAPPER_MAPPPING.get(task)
  if framework is None:
    framework = pipe_wrapper.default_framework
    if framework is None:
      raise RuntimeError(
        "Framework is not specified and the pipeline wrapper for {task} does not specify a default framework"
      )
    logger.info(f"Framework is not specified, use default framework {framework} for task {task}")

  if model is None:
    model = pipe_wrapper.default_model
    if model is None:
      raise ValueError(f"no default model for {task}")
    if "@" in model:
      model, revision = model.split("@")
      kwargs["revision"] = revision
    logger.info(f"Model is not specified, use default model {model}")
  
  if backend is None:
    backend = pipe_wrapper.default_backend
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
                            framework=framework,
                            backend=backend,
                            model_kwargs=model_kwargs,
                            **kwargs)
  pipe_wrapper.set_pipeline(pipeline)
  
  return pipe_wrapper
  