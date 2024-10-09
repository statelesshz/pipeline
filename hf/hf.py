import re
from functools import cached_property
from typing import Any, Dict, List, Union

from loguru import logger

from .util import Registry
from .hf_utils import pipeline_registry, hf_missing_package_error_message


task_cls_registry = Registry()


HF_DEFINED_TASKS = [
    # transformers
    "text-classification",
    "text-generation",
]


class HFPipelineWrapper:
    task: str = "undefined (please override this in your derived class)"
    framework: str = "pt"
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
        task = "text-classifciation"
        model_id = "/home/lynn/github/distilbert-base-uncsed"
        revision = None
        framework = "pt"
        return task, model_id, revision
    
    def __init__(self, name: str, model_str: str, framework: str):
        # if framework is not None就用这个
        task, model_id, revision = self._parse_model_str(model_str)

        self.model_id = model_id
        self.task = task
        self.revision = revision

        # 

    @property
    def metadata(self):
        res = super().metadata
        res.update({
            "task": self.task
        })
        return res
    
    @cached_property
    def pipeline(self):
        # TODO(lynn): checkme - get pipeline creator
        pipeline_creator = pipeline_registry.get("pt").get(self.task)
        if pipeline_creator is None:
            raise ValueError(f"Could not find pipeline creator for {self.task}")
        
        logger.info(
            f"Creating pipeline for {self.task}(model={self.model},"
            f" revision={self.revision}).\n"
            "HuggingFace download might take a while, please be patient..."
        )

        logger.info(
            "Note: HuggingFace caches the downloaded models in ~/.cache/huggingface/."
            " If you have already downloaded the model before, the download should be much"
            " faster. If you run out of disk space, you can delete the cache folder."
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
    
    # def init(self):
    #     super().init()
    #     # access pipeline here to trigger download and load
    #     self.pipeline

    def _run_pipeline(self, *args, **kwargs):
        import torch
        try:
            import torch_npu
        except Exception:
            ...
        
        # autocast causes invalid value (and generates black images) for text-to-image and image-to-image
        no_auto_cast_set = ("text-to-image", "image-to-image")
        if torch.npu.is_available() and self.task not in no_auto_cast_set:
            with torch.autocast(device_type="npu"):
                return self.pipeline(*args, **kwargs)
        elif torch.cuda.is_available() and self.task not in no_auto_cast_set:
            with torch.autocast(device_type="cuda"):
                return self.pipeline(*args, **kwargs)
        else:
            return self.pipeline(*args, **kwargs)

    @classmethod
    def create_from_model_str(cls, name, model_str):
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
        return task_cls(name, model_str)  # HFPipelieWrapper()


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

# HUGGING_FACE_SCHEMAS=["hf", "huggingface"]
# wrapper_registry = Registry()
# wrapper_registry.register(
#     HUGGING_FACE_SCHEMAS, HFPipelineWrapper.create_from_model_str
# )
