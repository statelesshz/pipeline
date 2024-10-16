import os.path
from typing import Optional, Literal, Type, Dict, Any

from loguru import logger

from openmind.utils.hub import OpenMindHub

from .base import BasePipelineWrapper
from .hf import (PipelineWrapper,
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
                 )
from .utils import get_task_from_readme


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

register_pipeline_wrapper(
  "visual-question-answering",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/blip_vqa_base@4450392",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "zero-shot-object-detection",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/owlvit_base_patch32@ff06496",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "zero-shot-classification",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/deberta_v3_large_zeroshot_v2.0@d38d6f4",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "depth-estimation",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/dpt_large@270fa97",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "image-to-image",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/swin2SR_classical_sr_x2_64@407e816",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "mask-generation",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/sam_vit_base@d0ad399",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "zero-shot-image-classification",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/siglip_so400m_patch14_384@b4099dd",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "feature-extraction",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/xlnet_base_cased@bc7408f",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "image-classification",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/beit_base_patch16_224@a46c2b5",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "image-to-text",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/blip-image-captioning-large@059b23b",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "text2text-generation",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/flan_t5_base@d15ab63",
    default_framework="pt",
    default_backend="transformers",
  )
)

register_pipeline_wrapper(
  "token-classification",
  PipelineWrapper(
    # <model_name>[@<revision>] or <model_name>
    default_model="PyTorch-NPU/camembert_ner@1390d33",
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
  