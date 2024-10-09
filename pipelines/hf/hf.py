from ..utils import Registry


task_cls_registry = Registry()


HF_DEFINED_TASKS = [
  # diffusers
  "text-to-image",
  # transformers
  "text-classification",
  "text-generation",
]


class 