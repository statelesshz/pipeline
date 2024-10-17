from ...utils import Registry, download_from_repo


pipeline_registry = Registry()


def create_mindformers_pipeline(task: str=None,
                                 model: str=None,
                                 tokenizer: str=None,
                                 image_processor: str=None,
                                 audio_processor: str = None,
                                 **kwargs):
  from mindformers import pipeline

  revision = kwargs.pop("revision", None)
  cache_dir = kwargs.pop("cache_dir", None)
  force_download = kwargs.pop("force_download", False)


  if isinstance(model, str):
    model_name_or_path = download_from_repo(
      model, revision=revision, cache_dir=cache_dir, force_download=force_download
    )
  else:
    model_name_or_path = model


  if isinstance(tokenizer, str):
    tokenizer_name_or_path = download_from_repo(
      tokenizer, revision=revision, cache_dir=cache_dir, force_download=force_download
    )
  else:
    tokenizer_name_or_path = tokenizer


  if isinstance(audio_processor, str):
    audio_processor_name_or_path = download_from_repo(
      audio_processor, revision=revision, cache_dir=cache_dir, force_download=force_download
    )
  else:
    audio_processor_name_or_path = audio_processor


  if isinstance(image_processor, str):
    image_processor_name_or_path = download_from_repo(
      image_processor, revision=revision, cache_dir=cache_dir, force_download=force_download
    )
  else:
    image_processor_name_or_path = image_processor

  pipe = pipeline(task=task,
                  model=model_name_or_path,
                  tokenizer = tokenizer_name_or_path,
                  image_processor = image_processor_name_or_path,
                  audio_processor = audio_processor_name_or_path
                  **kwargs)

  return pipe

for task in [
  "text-generation",
]:
  pipeline_registry.register(
    task,
    {
      "pt": {
        "mindformers": create_mindformers_pipeline,
      },
    }
  )