from loguru import logger

from ..utils import TRUST_REMOTE_CODE, Registry


pipeline_registry = Registry()


def create_transformers_pipeline(task, model, revision, **kwargs):
  from transformers import pipeline, is_torch_npu_available
  import torch
  try:
    import torch_npu
  except Exception:
    ...
  
  _kwargs = {}
  if TRUST_REMOTE_CODE:
    _kwargs["trust_remote_code"] = TRUST_REMOTE_CODE

  if torch.cuda.is_available() or is_torch_npu_available():
    torch_dtype = torch.float16
  else:
    torch_dtype = torch.float32
  _kwargs["torch_dtype"] = torch_dtype

  model_kwargs = {
    "low_cpu_mem_usage": True,
  }

  if "model" not in _kwargs:
    _kwargs["model_kwargs"] = model_kwargs

  if torch.cuda.is_available() or is_torch_npu_available:
    _kwargs["device"] = 0
  _kwargs.setdefault("model", model)
  kwargs.update(_kwargs)
  
  try:
    pipe = pipeline(task=task, revision=revision, **kwargs)
  except Exception as e:
    logger.info(
      f"Failed to create pipeline with {torch_dtype}: {e}, fallback to fp32"
    )
    if "low_cpu_mem_usage" in str(e).lower():
        logger.info(
            "error seems to be caused by low_cpu_mem_usage, retry without"
            " low_cpu_mem_usage"
        )
        kwargs.get("model_kwargs", {}).pop("low_cpu_mem_usage")
        if not kwargs.get("model_kwargs"):
            kwargs.pop("model_kwargs")
    # fallback to fp32
    kwargs.pop("torch_dtype")
    pipe = pipeline(task=task, revision=revision, **kwargs)
  return pipe

for task in [
  "text-generation",
]:
  pipeline_registry.register(
    task,
    {
      "pt": {
        "transformers": create_transformers_pipeline,
      },
    }
  )