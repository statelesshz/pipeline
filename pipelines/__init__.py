import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Literal, Union

from loguru import logger

from .base import task_cls_registry
from .hf.hf import HFPipelineWrapper, HF_DEFINED_TASKS


def get_pipeline_wrapper(task):
  if task in HF_DEFINED_TASKS:
    return HFPipelineWrapper


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
  return get_pipeline_wrapper(task).build(task, model, revision, framework, backend, **kwargs)
