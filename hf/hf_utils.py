from typing import List
from loguru import logger

from .util import Registry, TRUST_REMOTE_CODE


hf_no_attention_mask_models = {"microsoft/phi-1", "microsoft/phi-1_5"}

pipeline_registry = Registry()


def create_transformers_pipeline(task, model, revision):
    from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForCausalLM
    import torch
    try:
        import torch_npu
    except Exception:
        ...

    kwargs = {}
    if TRUST_REMOTE_CODE:
        kwargs["trust_remote_code"] = TRUST_REMOTE_CODE

    # TODO: check if bfloat16 is well supported
    kwargs["torch_dtype"] = torch.flaot32

    model_kwargs = {
        "low_cpu_mem_usage": True,
    }

    if "model" not in kwargs:
        kwargs["model_kwargs"] = model_kwargs

    if torch.npu.is_available() or torch.cuda.is_available():
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
        # fallback to fp32
        kwargs.pop("torch_dtype")
        pipe = pipeline(task=task, revision=revision, **kwargs)
    return pipe


for task in [
    "text-classification",
    "text_generation",
]:
    # TODO
    pipeline_registry.register(
        task,
        {"pt": create_transformers_pipeline},
    )


def hf_missing_package_error_message(
    pipeline_name: str, missing_packages: List[str]
) -> str:
    return (
        "HuggingFace reported missing packages for the specified pipeline. You can see"
        " the hf error message above. \n\nThis is not a bug of LeptonAI, as"
        " HuggingFace pipelines do not have a standard way to pre-determine and"
        " install dependencies yet. As a best-effort attempt, here are steps you can"
        " take to fix this issue:\n\n1. If you are running locally, you can install"
        " the missing packages with pip as follows:\n\npip install"
        f" {' '.join(missing_packages)}\n\n(note that some package names and pip names"
        " may be different, and you may need to search pypi for the correct package"
        " name)\n\n2. If you are using LeptonAI library, we maintain a mapping from"
        " known HuggingFace pipelines to their dependencies. We appreciate if you can"
        " send a PR to https://github.com/leptonai/leptonai/ to add the missing"
        " dependencies. please refer to"
        " https://github.com/leptonai/leptonai/blob/main/leptonai/photon/hf/hf_dependencies.py"
        " for more details."
    )
