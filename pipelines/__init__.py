from typing import Optional, Literal

from loguru import logger

from .utils import FrameWork


def pipeline(
  task: Optional[str] = None,
  model: Optional[str] = None,
  config: Optional[str] = None,
  tokenzier: Optional[str] = None,
  feature_extractor: Optional[str] = None,
  image_processor: Optional[str] = None,
  framework: Optional[FrameWork] = None,
  backend: Optional[str] = None,
  **kwargs,
):
  ...
