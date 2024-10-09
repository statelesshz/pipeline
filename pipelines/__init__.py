import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Literal, Union

from loguru import logger

from .base import task_cls_registry
from .hf.hf import HFPipelineBuilder


# 根据传入的model_str解析模型信息
class ModelInfo:

  def __init__(self, model_dir: Union[str, os.PathLike]):
    self.model_dir = model_dir
    self.is_local = os.path.isdir(model_dir)

  @lru_cache(maxsize=None)
  def list_files(self):
    if self.is_local:
      path = Path(self.model_dir)
      return [str(entry) for entry in path.rglob("*") if entry.is_file()]
    
  @property
  def framework(self):
    breakpoint()
    for file in self.list_files():
      # hf-transformers/diffusers可以通过这种方式判断？
      if file.endswith(".bin") or file.endswith(".safetensors"):
        return "pt"
    # mindspore侧框架信息如何判断？或者是否要约束pipeline接口一定要指定framework="ms"？
    return "ms"
  
  @property
  def backend(self):
    # backend:transformers/diffusers/mindformers
    if self.framework == "pt":
      # 注意我们需要先遍历一遍判断是否是diffusers
      for file in self.list_files():
        with open(file, "r", encoding="utf-8") as fp:
          config = json.load(fp)
        if "_diffusers_version" in config:
          return "diffusers"
      for file in self.list_files():
        with open(file, "r", encoding="utf-8") as fp:
          config = json.load(fp)
        if "transformers_version" in config:
          return "transformers"
    else:
      # TODO: for mindspore, just mock now
      return "mindformers"
  
  @property
  def metadata(self):
    return {
      "model_dir": self.model_dir,
      "is_local": self.is_local,
      "framework": self.framework,
      "backend": self.backend,
    }


def get_pipeline_builder(task, model, revision, framework, backend, **kwargs):
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
      " for debugging purpose: {model}"
    )
  return task_cls(task, model, revision, framework, backend, **kwargs)


# hf-transformers的pipeline接口用法非常灵活
# model可以是本地路径或者model id on hub也可以是具体对象
# 具体参数校验规则需要整理下
def pipeline(
  task: Optional[str] = None,
  model: Optional[str] = None,
  # config: Optional[str] = None,
  # tokenzier: Optional[str] = None,
  # feature_extractor: Optional[str] = None,
  # image_processor: Optional[str] = None,
  revision: Optional[str] = None,
  framework: Optional[Literal["pt", "ms"]] = None,
  backend: Optional[str] = None,
  **kwargs,
):
  # TODO: 完成必要的参数校验，保证传递给PipelineBuilder非空实参
  return HFPipelineBuilder.create(task, model, revision, framework, backend, **kwargs)
