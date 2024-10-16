import copy
from functools import cached_property
from typing import Callable, Optional, List, Union, Dict, Any

from loguru import logger

from openmind.utils import is_vision_available

from ..base import BasePipelineWrapper, PTBasePipeline
from .hf_utils import pipeline_registry

if is_vision_available():
    from PIL import Image


class PipelineWrapper(BasePipelineWrapper):
  def __init__(self,
               task: str = None,
               default_framework: str = None,
               default_model: str = None,
               default_backend: str = None,
               **kwargs):
    super().__init__(task, default_framework, default_model, default_backend, **kwargs)


class HFPipeline(PTBasePipeline):
  def __init__(self,
               task: str = None, 
               model: str = None,
               config: str = None, 
               tokenizer: str = None, 
               feature_extractor: str = None, 
               image_processor: str = None,
               framework: str = None, 
               backend: str = None,
               model_kwargs: Dict = None,
               **kwargs):
    self.task = task
    self.model = model
    self.config = config
    self.tokenizer =tokenizer
    self.feature_extractor = feature_extractor
    self.image_processor = image_processor
    self.framework = "pt"
    self.backend = backend
    self.model_kwargs = copy.deepcopy(model_kwargs)
    self.kwargs = copy.deepcopy(kwargs)

    # access pipeline here to trigger download and load
    self.pipeline

  @cached_property
  def pipeline(self) -> Callable:
    try:
      pipeline_creator = pipeline_registry.get(self.task).get(self.framework).get(self.backend)
    except Exception as e:
      # If any error occurs, we issue a warning, but don't exit immediately.
      logger.warning(
        "An error occurred while trying to get pipline creator. Error"
        f" details: {e}"
      )
      pipeline_creator = None
    if pipeline_creator is None:
      raise ValueError(f"Could not find pipeline creator for {self.task}:{self.framework}:{self.backend}")
    
    logger.info(
      f"Creating pipeline for {self.task}(framework={self.framework}, backend={self.backend},"
      f" model={self.model}, revision={self.revision}).\n"
      "openMind download might take a while, please be patient..."
    )

    try:
      # TODO: 确认这里要传入什么参数
      pipeline = pipeline_creator(
        task=self.task,
        model=self.model,
        config=self.config,
        tokenizer=self.tokenizer,
        feature_extractor=self.feature_extractor,
        image_processor=self.image_processor,
        model_kwargs=self.model_kwargs,
        **self.kwargs,
      )
    except ImportError as e:
      # TODO: add more user-friendly error messages
      raise e
    return pipeline

  def _run_pipeline(self, *args, **kwargs):
    # TODO: do we need this?
    try:
      import torch
      # from accelerate import
      from accelerate.utils import is_npu_available
      if is_npu_available():
        import torch_npu
    except ImportError as e:
      # TODO: add more user-friendly error messages
      raise e

    # autocast causes invalid value (and generates black images) for text-to-image and image-to-image
    no_auto_cast_set = ("text-to-image", "image-to-image")

    if torch.cuda.is_available() and self.task not in no_auto_cast_set:
      with torch.autocast(device_type="cuda"):
        return self.pipeline(*args, **kwargs)
    else:
      return self.pipeline(*args, **kwargs)
    return self.pipeline(*args, **kwargs)


def _get_generated_text(res):
    if isinstance(res, str):
        return res
    elif isinstance(res, dict):
        return res["generated_text"]
    elif isinstance(res, list):
        if len(res) == 1:
            return _get_generated_text(res[0])
        else:
            return [_get_generated_text(r) for r in res]
    else:
        raise ValueError(f"Unsupported result type in _get_generated_text: {type(res)}")


class TextGenerationPipeline(HFPipeline):
  def __init__(self,
               task: str = None, 
               model: str = None,
               config: str = None, 
               tokenizer: str = None, 
               feature_extractor: str = None, 
               image_processor: str = None,
               revision: str = None,
               framework: str = None, 
               backend: str = None,
               model_kwargs: Dict = None,
               **kwargs):
    self.task = task
    self.model = model
    self.config = config
    self.tokenizer =tokenizer
    self.feature_extractor = feature_extractor
    self.image_processor = image_processor
    self.revision = revision
    self.framework = "pt"
    self.backend = "transformers"
    self.model_kwargs = copy.deepcopy(model_kwargs)
    self.kwargs = copy.deepcopy(kwargs)

    # access pipeline here to trigger download and load
    self.pipeline
  
  def __call__(
    self,
    inputs: Union[str, List[str]],
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = 1.0,
    repetition_penalty: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    max_time: Optional[float] = None,
    return_full_text: bool = True,
    num_return_sequences: int = 1,
    do_sample: bool = True,
    **kwargs,
  ) -> Union[str, List[str]]:
    res = self._run_pipeline(
      inputs,
      top_k=top_k,
      top_p=top_p,
      temperature=temperature,
      repetition_penalty=repetition_penalty,
      max_new_tokens=max_new_tokens,
      max_time=max_time,
      return_full_text=return_full_text,
      num_return_sequences=num_return_sequences,
      do_sample=do_sample,
      **kwargs,
    )

    return _get_generated_text(res)


class VisualQuestionAnsweringPipeline(HFPipeline):
  def __init__(self,
               task: str = None,
               model: str = None,
               config: str = None,
               tokenizer: str = None,
               feature_extractor: str = None,
               image_processor: str = None,
               revision: str = None,
               framework: str = None,
               backend: str = None,
               model_kwargs: Dict = None,
               **kwargs):
    self.task = task
    self.model = model
    self.config = config
    self.tokenizer = tokenizer
    self.feature_extractor = feature_extractor
    self.image_processor = image_processor
    self.revision = revision
    self.framework = "pt"
    self.backend = "transformers"
    self.model_kwargs = copy.deepcopy(model_kwargs)
    self.kwargs = copy.deepcopy(kwargs)

    # access pipeline here to trigger download and load
    self.pipeline

  def __call__(
          self,
          image: Union[str, List[str]],
          question: Union[str, List[str]] = None,
          **kwargs,
  ):
    res = self._run_pipeline(
      image=image,
      question=question,
      **kwargs,
    )

    return res


class ZeroShotObjectDetectionPipeline(HFPipeline):
  def __init__(self,
               task: str = None,
               model: str = None,
               config: str = None,
               tokenizer: str = None,
               feature_extractor: str = None,
               image_processor: str = None,
               revision: str = None,
               framework: str = None,
               backend: str = None,
               model_kwargs: Dict = None,
               **kwargs):
    self.task = task
    self.model = model
    self.config = config
    self.tokenizer = tokenizer
    self.feature_extractor = feature_extractor
    self.image_processor = image_processor
    self.revision = revision
    self.framework = "pt"
    self.backend = "transformers"
    self.model_kwargs = copy.deepcopy(model_kwargs)
    self.kwargs = copy.deepcopy(kwargs)

    # access pipeline here to trigger download and load
    self.pipeline

  def __call__(
          self,
          image: Union[str, "Image.Image", List[Dict[str, Any]]],
          candidate_labels: Union[str, List[str]] = None,
          **kwargs,
  ):
    res = self._run_pipeline(
      image=image,
      candidate_labels=candidate_labels,
      **kwargs,
    )

    return res


class ZeroShotClassificationPipeline(HFPipeline):
  def __init__(self,
               task: str = None,
               model: str = None,
               config: str = None,
               tokenizer: str = None,
               feature_extractor: str = None,
               image_processor: str = None,
               revision: str = None,
               framework: str = None,
               backend: str = None,
               model_kwargs: Dict = None,
               **kwargs):
    self.task = task
    self.model = model
    self.config = config
    self.tokenizer = tokenizer
    self.feature_extractor = feature_extractor
    self.image_processor = image_processor
    self.revision = revision
    self.framework = "pt"
    self.backend = "transformers"
    self.model_kwargs = copy.deepcopy(model_kwargs)
    self.kwargs = copy.deepcopy(kwargs)

    # access pipeline here to trigger download and load
    self.pipeline

  def __call__(
          self,
          sequences: Union[str, List[str]],
          *args,
          **kwargs,
  ):
    res = self._run_pipeline(
      sequences=sequences,
      **kwargs,
    )

    return res


class DepthEstimationPipeline(HFPipeline):
  def __init__(self,
               task: str = None,
               model: str = None,
               config: str = None,
               tokenizer: str = None,
               feature_extractor: str = None,
               image_processor: str = None,
               revision: str = None,
               framework: str = None,
               backend: str = None,
               model_kwargs: Dict = None,
               **kwargs):
    self.task = task
    self.model = model
    self.config = config
    self.tokenizer = tokenizer
    self.feature_extractor = feature_extractor
    self.image_processor = image_processor
    self.revision = revision
    self.framework = "pt"
    self.backend = "transformers"
    self.model_kwargs = copy.deepcopy(model_kwargs)
    self.kwargs = copy.deepcopy(kwargs)

    # access pipeline here to trigger download and load
    self.pipeline

  def __call__(
          self,
          images: Union[str, List[str], "Image.Image", List["Image.Image"]] = None,
          **kwargs,
  ):
    res = self._run_pipeline(
      images=images,
      **kwargs,
    )

    return res


class ImageToImagePipeline(HFPipeline):
  def __init__(self,
               task: str = None,
               model: str = None,
               config: str = None,
               tokenizer: str = None,
               feature_extractor: str = None,
               image_processor: str = None,
               revision: str = None,
               framework: str = None,
               backend: str = None,
               model_kwargs: Dict = None,
               **kwargs):
    self.task = task
    self.model = model
    self.config = config
    self.tokenizer = tokenizer
    self.feature_extractor = feature_extractor
    self.image_processor = image_processor
    self.revision = revision
    self.framework = "pt"
    self.backend = "transformers"
    self.model_kwargs = copy.deepcopy(model_kwargs)
    self.kwargs = copy.deepcopy(kwargs)

    # access pipeline here to trigger download and load
    self.pipeline

  def __call__(
          self,
          images: Union[str, List[str], "Image.Image", List["Image.Image"]] = None,
          **kwargs,
  ) -> Union["Image.Image", List["Image.Image"]]:
    res = self._run_pipeline(
      images=images,
      **kwargs,
    )

    return res


class MaskGenerationPipeline(HFPipeline):
  def __init__(self,
               task: str = None,
               model: str = None,
               config: str = None,
               tokenizer: str = None,
               feature_extractor: str = None,
               image_processor: str = None,
               revision: str = None,
               framework: str = None,
               backend: str = None,
               model_kwargs: Dict = None,
               **kwargs):
    self.task = task
    self.model = model
    self.config = config
    self.tokenizer = tokenizer
    self.feature_extractor = feature_extractor
    self.image_processor = image_processor
    self.revision = revision
    self.framework = "pt"
    self.backend = "transformers"
    self.model_kwargs = copy.deepcopy(model_kwargs)
    self.kwargs = copy.deepcopy(kwargs)

    # access pipeline here to trigger download and load
    self.pipeline

  def __call__(
          self,
          image,
          *args,
          num_workers=None,
          batch_size=None,
          **kwargs
  ):
    res = self._run_pipeline(
      image=image,
      *args,
      num_works=num_workers,
      batch_size=batch_size,
      **kwargs,
    )

    return res