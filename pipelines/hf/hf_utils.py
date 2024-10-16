from typing import Dict
from openmind import AutoTokenizer
from ..utils import Registry, download_from_repo, get_task_from_readme


pipeline_registry = Registry()


def create_transformers_pipeline(task: str=None,
                                 model: str=None,
                                 config: str=None,
                                 tokenizer: str=None,
                                 feature_extractor: str=None,
                                 image_processor: str=None,
                                 model_kwargs: Dict=None,
                                 **kwargs):
  from transformers import pipeline

  revision = kwargs.pop("revision", None)
  cache_dir = kwargs.pop("cache_dir", None)
  force_download = kwargs.pop("force_download", False)
  use_fast = kwargs.pop("use_fast", False)
  device = kwargs.pop("device", None)
  device_map = kwargs.pop("device_map", None)
  torch_dtype = kwargs.pop("torch_dtype", None)
  use_auth_token = kwargs.pop("use_auth_token", None)
  trust_remote_code = kwargs.pop("trust_remote_code", None)
  _commit_hash = kwargs.get("_commit_hash", None)

  if isinstance(model, str):
    model_name_or_path = download_from_repo(
      model, revision=revision, cache_dir=cache_dir, force_download=force_download
    )
  else:
    model_name_or_path = model

  if tokenizer is not None:
    if isinstance(tokenizer, str):
      tokenizer_name_or_path = download_from_repo(
        tokenizer, revision=revision, cache_dir=cache_dir, force_download=force_download
      )
    else:
      tokenizer_name_or_path = tokenizer
  else:
    if (task == "text-generation" or task == "text_generation") and isinstance(model, str):
      tokenizer_kwargs = {
        "revision": revision,
        "token": use_auth_token,
        "trust_remote_code": trust_remote_code,
        "_commit_hash": _commit_hash,
        "torch_dtype": torch_dtype,
      }
      tokenizer_name_or_path = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
    else:
      tokenizer_name_or_path = tokenizer

  if isinstance(config, str):
    config_name_or_path = download_from_repo(
      config, revision=revision, cache_dir=cache_dir, force_download=force_download
    )
  else:
    config_name_or_path = config

  if isinstance(feature_extractor, str):
    feature_extractor_name_or_path = download_from_repo(
      feature_extractor, revision=revision, cache_dir=cache_dir, force_download=force_download
    )
  else:
    feature_extractor_name_or_path = feature_extractor

  if isinstance(image_processor, str):
    image_processor_name_or_path = download_from_repo(
      image_processor, revision=revision, cache_dir=cache_dir, force_download=force_download
    )
  else:
    image_processor_name_or_path = image_processor

    pipe = pipeline(task=task,
                    model=model_name_or_path,
                    tokenizer = tokenizer_name_or_path,
                    config = config_name_or_path,
                    feature_extractor = feature_extractor_name_or_path,
                    image_processor = image_processor_name_or_path,
                    revision = revision,
                    use_fast=use_fast,
                    device = device,
                    device_map = device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    model_kwargs= model_kwargs,
                    **kwargs)

  return pipe

for task in [
  "text-generation",
  "visual-question-answering",
  "zero-shot-object-detection",
  "zero-shot-classification",
  "depth-estimation",
  "image-to-image",
  "mask-generation",
]:
  pipeline_registry.register(
    task,
    {
      "pt": {
        "transformers": create_transformers_pipeline,
      },
    }
  )