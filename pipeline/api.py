import os
import json
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Optional, Union

from .base import FrameWork, PIPELINE_WRAPPER_MAPPING, PIPELINE_MAPPING


MODEL_TASK_MAPPING = OrderedDict(
  [
    ("qwen2", "chat"),
    ("stable-diffusion", "text-to-image"),
  ]
)

def _find_config_json(model_str):
  p = Path(model_str)
  if not p.is_dir():
    print(f"The specified path is not a directory or does not exist: {model_str}")
    return None
  for entry in p.iterdir():
    if entry.is_file() and entry.name == "config.json":
      return entry
  return None


def read_config_json(model_str):
  config_file_path = _find_config_json(model_str)
  try:
    with open(config_file_path, "r", encoding="utf-8") as file:
      config = json.load(file)
      return config
  except Exception as e:
    print(f"An unknown error occurred while opening the file: {e}")
    return None


def parse_model_metadata(model_str: str):
  # 尽力而为的解析
  is_local = os.path.isdir(model_str)

  if is_local:
    print("start scanning local file.")
    # 1. 判断框架

    # fetch the content in config.json
    config = read_config_json(model_str)
    model_type = config.get("model_type", "unknown_model_type")
    task = MODEL_TASK_MAPPING.get(model_type, "unknown_task")
    return task
  else:
    # TODO
    print("get model metadata from hub")
    # model_repo
    return None


# now, only support transformers diffusers
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
    task = parse_model_metadata(model)
  
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

