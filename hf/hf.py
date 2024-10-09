import re
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .util import Registry
from .hf_utils import pipeline_registry, hf_missing_package_error_message


task_cls_registry = Registry()


HF_DEFINED_TASKS = [
    # diffusers
    "text-to-image",
    # transformers
    "text-classification",
    "text-generation",
]


class HFPipelineWrapper:
    task: str = "undefined (please override this in your derived class)"
    # framework: str = "undefined (please override this in your derived class)"
    model_id: str = "undefined (please override this in your derived class)"
    # requirement_dependency: Optional[List[str]] = []

    def __init_subclass__(cls, **kwargs) -> None:
        if cls.task not in HF_DEFINED_TASKS:
            raise ValueError(
                f"You made a programming error: the task {cls.task} is not a"
                " supported task defined in HuggingFace. If you believe this is an"
                " error, please file an issue."
            )
        task_cls_registry.register(cls.task, cls)

    @classmethod
    def supported_tasks(cls):
        """
        Returns the set of supported tasks.
        """
        return task_cls_registry.keys()
    
    @classmethod
    def _parse_model_str(cls, model_str):
        # model_str
        # hf:<model_name>[@<revision>]
        # hf:<task_name>:<model_name>[@<revision>]
        # mock
        # task = "text-classification"
        task = "text-to-image"
        model_id = model_str
        revision = None
        # framework = "pt"
        return task, model_id, revision
    
    def __init__(self, model_str: str):
        task, model_id, revision = self._parse_model_str(model_str)

        self.model = model_id
        self.task = task
        self.revision = revision

        self.pipeline
    
    @cached_property
    def pipeline(self):
        # TODO(lynn): checkme - get pipeline creator
        pipeline_creator = pipeline_registry.get(self.task).get("pt") # 可能会报错 None.get(self.task) -> NoneType object has no attribute get
        if pipeline_creator is None:
            raise ValueError(f"Could not find pipeline creator for {self.task}")
        
        logger.info(
            f"Creating pipeline for {self.task}(model={self.model},"
            f" revision={self.revision}).\n"
            "HuggingFace download might take a while, please be patient..."
        )
        try:
            pipeline = pipeline_creator(
                task=self.task,
                model=self.model,
                revision=self.revision,
            )
        except ImportError as e:
            # Huggingface has a mechanism that detects dependencies, and then prints dependent
            # libraries in the error message. When this happens, we want to parse the error and
            # then tell the user what they should do.
            # See https://github.com/huggingface/transformers/blob/ce2e7ef3d96afaf592faf3337b7dd997c7ad4928/src/transformers/dynamic_module_utils.py#L178
            # for the source code that prints the error message.
            pattern = (
                "This modeling file requires the following packages that were not found"
                " in your environment: (.*?). Run `pip install"
            )
            match = re.search(pattern, e.msg)
            if match:
                missing_packages = match.group(1).split(", ")
                raise ImportError(
                    hf_missing_package_error_message(self.task, missing_packages)
                ) from e
            else:
                raise e
        return pipeline

    def _run_pipeline(self, *args, **kwargs):
        import torch
        try:
            import torch_npu
        except Exception:
            ...
        
        # autocast causes invalid value (and generates black images) for text-to-image and image-to-image
        no_auto_cast_set = ("text-to-image", "image-to-image")
        # if torch.npu.is_available() and self.task not in no_auto_cast_set:
        #     with torch.autocast(device_type="npu"):
        #         return self.pipeline(*args, **kwargs)
        if torch.cuda.is_available() and self.task not in no_auto_cast_set:
            with torch.autocast(device_type="cuda"):
                return self.pipeline(*args, **kwargs)
        else:
            return self.pipeline(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    # 入口
    @classmethod
    def create_from_model_str(cls, model_str):
        task, _, _ = cls._parse_model_str(model_str)
        task_cls = task_cls_registry.get(task)
        if task_cls is None:
            raise ValueError(
                f"openmind currently does not support the specified task: {task}. If"
                " you would like us to support this task, please let us know by"
                " opening an issue"
                " at https://xxx, and"
                " kindly include the specific model that you are trying to run for"
                " debugging purposes: {model_str}"
            )
        # model_str
        # hf:<model_name>[@<revision>]
        # hf:<task_name>:<model_name>[@<revision>]
        return task_cls(model_str)  # HFPipelieWrapper()


class HFTextClassificationWrapper(HFPipelineWrapper):
    task = "text-classification"

    def __call__(
        self,
        inputs: Union[str, List[str]],
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        res = self._run_pipeline(
            inputs,
            **kwargs,
        )
        return res


class HFTextToImageWrapper(HFPipelineWrapper):
    task = "text-to-image"

    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        seed: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        import torch
        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        
        if seed is not None:
            if not isinstance(seed, list):
                seed = [seed]
            generator = [
                torch.Generator(device=self._device).manual_seed(s) for s in seed
            ]
        else:
            generator = None

        return self._run_pipeline(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            **kwargs,
        ).images[0]
