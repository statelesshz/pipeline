from typing import Optional, Literal

from .builder import get_pipeline_wrapper


def pipeline(
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
  return get_pipeline_wrapper(task, 
                              model, 
                              config, 
                              tokenizer, 
                              feature_extractor, 
                              image_processor,
                              revision,
                              framework, 
                              backend, 
                              **kwargs)