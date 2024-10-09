import os
import json
from functools import lru_cache
from pathlib improt Path
from typing import Any, Union

from loguru import logger


TRUST_REMOTE_CODE = os.environ.get("OM_TRUST_REMOTE_CODE", "true").lower() in (
    "true",
    "1",
    "t",
    "on",
)


# 根据传入的model_str解析模型信息
class ModelInfo:

  def __init__(self, model_dir: Union[str, os.PathLike]):
    self.model_dir = model_dir
    self.is_local = os.path.isdir(model_dir)

  @lru_cache(maxsize=None)
  def list_files(self):
    if self.is_local:
      path = Path(self.model_dir)
      return [str(entry) for entry in path.rglob("*") if entry.is_file()]
    
  @property
  def framework(self):
    breakpoint()
    for file in self.list_files():
      # hf-transformers/diffusers可以通过这种方式判断？
      if file.endswith(".bin") or file.endswith(".safetensors"):
        return "pt"
    # mindspore侧框架信息如何判断？或者是否要约束pipeline接口一定要指定framework="ms"？
    return "ms"
  
  @property
  def backend(self):
    # backend:transformers/diffusers/mindformers
    if self.framework == "pt":
      # 注意我们需要先遍历一遍判断是否是diffusers
      for file in self.list_files():
        with open(file, "r", encoding="utf-8") as fp:
          config = json.load(fp)
        if "_diffusers_version" in config:
          return "diffusers"
      for file in self.list_files():
        with open(file, "r", encoding="utf-8") as fp:
          config = json.load(fp)
        if "transformers_version" in config:
          return "transformers"
    else:
      # TODO: for mindspore, just mock now
      return "mindformers"
  
  @property
  def metadata(self):
    return {
      "model_dir": self.model_dir,
      "is_local": self.is_local,
      "framework": self.framework,
      "backend": self.backend,
    }



class Registry:
    """
    A utility class to register and retrieve values by keys.
    """

    def __init__(self):
        self._map = {}

    def register(self, keys: Any, value: Any):
        try:
            _ = iter(keys)
            is_iterable = True
        except TypeError:
            is_iterable = False

        if isinstance(keys, str) or not is_iterable:
            keys = [keys]

        for key in keys:
            if key in self._map:
                logger.warning(
                    f'Overriding previously registered "{key}" value "{value}"'
                )
            self._map[key] = value

    def get(self, key: Any) -> Any:
        if key in self._map:
            return self._map[key]
        return None

    def keys(self):
        return self._map.keys()
