import copy
from functools import cached_property
from typing import Callable, Optional, List, Union, Dict, Any

from loguru import logger

from openmind.utils import is_vision_available

from ...base import PTBasePipeline
from .hf_utils import pipeline_creator_registry

if is_vision_available():
    from PIL.Image import Image


class HFPipeline(PTBasePipeline):
  backend: str = "transformers"
  requirement_dependency = [
    "accelerate",
    "transformers",
  ]

  def __init__(self,
               model: str = None,
               config: str = None, 
               tokenizer: str = None, 
               feature_extractor: str = None, 
               image_processor: str = None,
               model_kwargs: Dict = None,
               **kwargs):
    self.model = model
    self.config = config
    self.tokenizer =tokenizer
    self.feature_extractor = feature_extractor
    self.image_processor = image_processor
    self.model_kwargs = copy.deepcopy(model_kwargs)
    
    self.kwargs = copy.deepcopy(kwargs)
    self.task = self.kwargs.pop("task", None)
    self.framework = self.kwargs.pop("framework", None)
    self.backend = self.kwargs.pop("backend", None)

    # check dependencies
    self.check_dependency()

    # access pipeline here to trigger download and load
    self.pipeline

  @cached_property
  def pipeline(self) -> Callable:
    try:
      pipeline_creator = pipeline_creator_registry.get(self.task).get(self.framework).get(self.backend)
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
      f" model={self.model}, revision={self.kwargs.get('revision')}).\n"
      "openMind download might take a while, please be patient..."
    )

    try:
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

    # autocast causes invalid value (and generates black images) for text-to-image
    no_auto_cast_set = ("text-to-image")

    if torch.cuda.is_available() and self.task not in no_auto_cast_set:
      with torch.autocast(device_type="cuda"):
        return self.pipeline(*args, **kwargs)
    else:
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
  task = "text-generation"
  
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
  task = "visual-question-answering"

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
  task = "zero-shot-object-detection"
  requirement_dependency = ["pillow"]

  def __call__(
          self,
          image: Union[str, "Image", List[Dict[str, Any]]],
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
  task = "zero-shot-classification"

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
  task = "depth-estimation"
  requirement_dependency = ["pillow"]

  def __call__(
          self,
          images: Union[str, List[str], "Image", List["Image"]] = None,
          **kwargs,
  ):
    res = self._run_pipeline(
      images=images,
      **kwargs,
    )

    return res


class ImageToImagePipeline(HFPipeline):
  task = "image-to-image"
  requirement_dependency = ["pillow"]

  def __call__(
          self,
          images: Union[str, List[str], "Image", List["Image"]] = None,
          **kwargs,
  ) -> Union["Image", List["Image"]]:
    res = self._run_pipeline(
      images=images,
      **kwargs,
    )

    return res


class MaskGenerationPipeline(HFPipeline):
  task = "mask-generation"

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


class ZeroShotImageClassificationPipeline(HFPipeline):
  task = "zero-shot-image-classification"
  requirement_dependency = ["pillow"]
 
  def __call__(
    self,
    images: Union[str, List[str], "Image", List["Image"]],
    **kwargs,
  ) -> Union[str, List[str]]:
    res = self._run_pipeline(
      images,
      **kwargs,
    )

    return res


class FeatureExtractionPipeline(HFPipeline):
  task = "feature-extraction"

  def __call__(
    self,
    *args,
    **kwargs,
  ) -> Union[str, List[str]]:
    res = self._run_pipeline(
      *args,
      **kwargs,
    )

    return res


class ImageClassificationPipeline(HFPipeline):
  task = "image-classification"
  requirement_dependency = ["pillow"]

  def __call__(
    self,
    images: Union[str, List[str], "Image", List["Image"]],
    **kwargs,
  ) -> Union[str, List[str]]:
    res = self._run_pipeline(
      images,
      **kwargs,
    )

    return res


class ImageToTextPipeline(HFPipeline):
  task = "image-to-text"
  requirement_dependency = ["pillow"]

  def __call__(
    self,
    images: Union[str, List[str], "Image", List["Image"]],
    **kwargs,
  ) -> Union[str, List[str]]:
    res = self._run_pipeline(
      images,
      **kwargs,
    )

    return res


class Text2TextGenerationPipeline(HFPipeline):
  task = "text2text-generation"

  def __call__(
    self,
    *args,
    **kwargs,
  ) -> Union[str, List[str]]:
    res = self._run_pipeline(
      *args,
      **kwargs,
    )

    return res


class TokenClassificationPipeline(HFPipeline):
  task = "token-classification"

  def __call__(
    self,
    inputs: Union[str, List[str]],
    **kwargs,
  ) -> Union[str, List[str]]:
    res = self._run_pipeline(
      inputs,
      **kwargs,
    )

    return res


class FillMaskPipeline(HFPipeline):
  task = "fill-mask"

  def __call__(
          self,
          inputs,
          *args,
          targets=None,
          top_k=None,
          **kwargs
  ):
    res = self._run_pipeline(
      inputs=inputs,
      *args,
      targets=targets,
      top_k=top_k,
      **kwargs,
    )

    return res


class QuestionAnsweringPipeline(HFPipeline):
  task = "question-answering"

  def __call__(
          self,
          question,
          context,
          top_k=1,
          doc_stride=128,
          max_answer_len=15,
          max_seq_len=384,
          max_question_len=64,
          handle_impossible_answer=False,
          align_to_words=True,
          **kwargs
  ):
    res = self._run_pipeline(
      question=question,
      context=context,
      top_k=top_k,
      doc_stride=doc_stride,
      max_answer_len=max_answer_len,
      max_seq_len=max_seq_len,
      max_question_len=max_question_len,
      handle_impossible_answer=handle_impossible_answer,
      align_to_words=align_to_words,
      **kwargs,
    )

    return res


class SummarizationPipeline(HFPipeline):
  task = "summarization"

  def __call__(
          self,
          inputs,
          return_text=True,
          return_tensors=False,
          clean_up_tokenization_spaces=False,
          **kwargs
  ):
    res = self._run_pipeline(
      inputs,
      return_text=return_text,
      return_tensors=return_tensors,
      clean_up_tokenization_spaces=clean_up_tokenization_spaces,
      **kwargs,
    )

    return res


class TableQuestionAnsweringPipeline(HFPipeline):
  task = "table-question-answering"

  def __call__(
          self,
          table,
          query,
          sequential=False,
          padding=False,
          truncation=False,
          **kwargs
  ):
    res = self._run_pipeline(
      table=table,
      query=query,
      sequential=sequential,
      padding=padding,
      truncation=truncation,
      **kwargs,
    )

    return res


class TranslationPipeline(HFPipeline):
  task = "translation"

  def __call__(
          self,
          *args,
          return_tensors=False,
          return_text=True,
          clean_up_tokenization_spaces=False,
          src_lang=None,
          tgt_lang=None,
          **kwargs
  ):
    res = self._run_pipeline(
      *args,
      return_tensors=return_tensors,
      return_text=return_text,
      clean_up_tokenization_spaces=clean_up_tokenization_spaces,
      src_lang=src_lang,
      tgt_lang=tgt_lang,
      **kwargs
    )

    return res


class TextClassificationPipeline(HFPipeline):
  task = "text-classification"

  def __call__(
          self,
          inputs,
          top_k=1,
          function_to_apply=None,
          **kwargs
  ):
    res = self._run_pipeline(
      inputs=inputs,
      top_k=top_k,
      function_to_apply=function_to_apply,
      **kwargs
    )

    return res
