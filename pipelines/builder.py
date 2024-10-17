from typing import Optional, Literal, Type, Dict, Any

from loguru import logger


from .base import pipeline_wrapper_registry
from .common.hf import (
  TextGenerationPipeline,
  VisualQuestionAnsweringPipeline,
  ZeroShotObjectDetectionPipeline,
  ZeroShotClassificationPipeline,
  DepthEstimationPipeline,
  ImageToImagePipeline,
  MaskGenerationPipeline,
  ZeroShotImageClassificationPipeline,
  FeatureExtractionPipeline,
  ImageClassificationPipeline,
  ImageToTextPipeline,
  Text2TextGenerationPipeline,
  TokenClassificationPipeline,
  FillMaskPipeline,
  QuestionAnsweringPipeline,
  SummarizationPipeline,
  TableQuestionAnsweringPipeline,
  TranslationPipeline,
  TextClassificationPipeline,
)
from .common.ms import TextGenerationPipeline as MSTextGenerationPipeline
from .utils import get_task_from_readme


PIPELINE_MAPPING = {}


def register_pipeline(task: str, framework: str, backend: str, pipeline: Type, **kwargs):
  if task not in PIPELINE_MAPPING:
    PIPELINE_MAPPING[task] = {}
  if framework not in PIPELINE_MAPPING[task]:
    PIPELINE_MAPPING[task][framework] = {}
  PIPELINE_MAPPING[task][framework][backend] = pipeline

register_pipeline(
  "text-generation",
  "pt",
  "transformers",
  TextGenerationPipeline
)

register_pipeline(
  "visual-question-answering",
  "pt",
  "transformers",
  VisualQuestionAnsweringPipeline
)

register_pipeline(
  "zero-shot-object-detection",
  "pt",
  "transformers",
  ZeroShotObjectDetectionPipeline
)

register_pipeline(
  "zero-shot-classification",
  "pt",
  "transformers",
  ZeroShotClassificationPipeline
)

register_pipeline(
  "depth-estimation",
  "pt",
  "transformers",
  DepthEstimationPipeline
)

register_pipeline(
  "image-to-image",
  "pt",
  "transformers",
  ImageToImagePipeline
)

register_pipeline(
  "mask-generation",
  "pt",
  "transformers",
  MaskGenerationPipeline
)

register_pipeline(
  "zero-shot-image-classification",
  "pt",
  "transformers",
  ZeroShotImageClassificationPipeline
)

register_pipeline(
  "feature-extraction",
  "pt",
  "transformers",
  FeatureExtractionPipeline
)

register_pipeline(
  "image-classification",
  "pt",
  "transformers",
  ImageClassificationPipeline
)

register_pipeline(
  "image-to-text",
  "pt",
  "transformers",
  ImageToTextPipeline
)

register_pipeline(
  "text2text-generation",
  "pt",
  "transformers",
  Text2TextGenerationPipeline
)

register_pipeline(
  "token-classification",
  "pt",
  "transformers",
  TokenClassificationPipeline
)

register_pipeline(
  "fill-mask",
  "pt",
  "transformers",
  FillMaskPipeline
)

register_pipeline(
  "question-answering",
  "pt",
  "transformers",
  QuestionAnsweringPipeline
)

register_pipeline(
  "summarization",
  "pt",
  "transformers",
  SummarizationPipeline
)

register_pipeline(
  "table-question-answering",
  "pt",
  "transformers",
  TableQuestionAnsweringPipeline
)

register_pipeline(
  "translation",
  "pt",
  "transformers",
  TranslationPipeline
)

register_pipeline(
  "text-classification",
  "pt",
  "transformers",
  TextClassificationPipeline
)

register_pipeline(
  "text-generation",
  "ms",
  "mindformers",
  MSTextGenerationPipeline
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

  return pipe_wrapper_cls(pipeline)

  