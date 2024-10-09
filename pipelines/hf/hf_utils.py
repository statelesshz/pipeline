from loguru import logger

from ..utils import TRUST_REMOTE_CODE, Registry


pipeline_registry = Registry()

# TODO: 是否需要增加**kwargs
def create_diffusion_pipeline(task, model, revision):
  try:
    import torch
    from diffusers import AutoPipelineForText2Image
  except ImportError:
    raise RuntimeError(
      "openmind pipeline requires torch and diffusers but they are not"
      " installed. Please install them with: pip install diffusers..."
    )
  
  if torch.cuda.is_available():
    torch_dtype = torch.float16
  else:
    torch_dtype = torch.bfloat16

  try:
    pipeline = AutoPipelineForText2Image.from_pretrained(
      model,
      revision=revision,
      torch_dtype=torch_dtype,
    )
  except Exception as e:
    logger.info(
      f"Failed to create pipeline with {torch_dtype}: {e}, fallback to fp32"
    )
    pipeline = AutoPipelineForText2Image.from_pretrained(
      model,
      revision=revision
    )
  if torch.cuda.is_available():
    pipeline = pipeline.to("cuda")
  return pipeline


# 不同库如transformers/diffusers都会支持相同的任务text-to-image
# 不同框架也会支持相同的任务，pt(diffusers)、ms(mindone)
for task in [
  "text-to-image",
]:
  pipeline_registry.register(
    task,
    {
      "pt": {
        "diffusers": create_diffusion_pipeline,
      },
    }
  )


# TODO: 是否需要增加**kwargs
def create_transformers_pipeline(task, model, revision):
  from transformers import pipeline
  import torch
  try:
    import torch_npu
  except Exception:
    ...
  
  kwargs = {}
  if TRUST_REMOTE_CODE:
    kwargs["trust_remote_code"] = TRUST_REMOTE_CODE

  if torch.cuda.is_available():
    torch_dtype = torch.float16
  else:
    torch_dtype = torch.float32
  

  kwargs["torch_dtype"] = torch_dtype

  model_kwargs = {
    "low_cpu_mem_usage": True,
  }

  if "model" not in kwargs:
    kwargs["model_kwargs"] = model_kwargs
  
  if torch.cuda.is_available():
    kwargs["device"] = 0
  kwargs.setdefault("model", model)
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
  "text-classification",
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
