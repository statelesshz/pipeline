import os
import json
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Optional, Union

from .utils import ModelInfo
from .base import FrameWork, PIPELINE_WRAPPER_MAPPING, PIPELINE_MAPPING


# # 需要维护一份model_type->task的映射表
# MODEL_TASK_MAPPING = OrderedDict(
#   [
#     ("qwen2", "chat"),
#     ("stable-diffusion", "text-to-image"),
#   ]
# )


# 跟transformers保持一致
def pipeline(
  task: Optional[str] = None,
  model: Optional[str] = None,
  config: Optional[str] = None,
  tokenizer: Optional[str] = None,
  feature_extractor: Optional[str] = None,
  image_processor: Optional[str] = None,
  framework: Optional[Literal["pt", "ms"]] = None,
  **kwargs,
):
  if task is None and model is not None:
    # parse task
    # task = parse_model_metadata(model)
    metadata = ModelInfo(model).metadata
    task = metadata["pipeline_tag"]
    if task is None:
      raise AttributeError(
        f'Unsupported model: "{model}" (the model did not'
        " specify a task).\nUsually, this means that the model creator did"
        " not intend to publish the model as a pipeline, and is only using"
        " HuggingFace Hub as a storage for the model weights and misc"
        " files. Thus, it is not possible to run the model automatically."
      )
  
  pipeline_wrapper = PIPELINE_WRAPPER_MAPPING[task]
  if framework is None and pipeline_wrapper.default_framework:
    framework = pipeline_wrapper.default_framework
  elif type(framework) is str:
    framework = FrameWork[framework.upper()]
  else:
    framework = FrameWork.PT

  if model is None:
    model = pipeline_wrapper.default_model
    if model is None:
      raise ValueError(f"no default model for {task}")
  if framework not in PIPELINE_MAPPING[task]:
    raise ValueError(f"{framework} not support for {task}")
  
  pipeline = PIPELINE_MAPPING[task][framework]
  print(f"current pipeline wrapper is: {pipeline_wrapper.__class__.__name__}")
  print(f"current pipeline is: {pipeline.__class__.__name__}")

  pipeline.init_and_load(model, config, tokenizer, feature_extractor, image_processor, **kwargs)
  pipeline_wrapper.set_pipeline(pipeline)
  return pipeline_wrapper
