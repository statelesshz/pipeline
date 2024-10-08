import re
import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union


# TODO: 实现hub上模型信息解析&mindformers模型解析
class ModelInfo:
    def __init__(self, model_dir: Union[str, os.PathLike]):
        self.model_dir = model_dir
        self.is_local = os.path.isdir(model_dir)
    
    @lru_cache(maxsize=None)
    def list_files(self):
      if self.is_local:
        path = Path(self.model_dir)
        return [str(entry) for entry in path.rglob("*") if entry.is_file()]
      else:
        # TODO: list files from_hub
        return []
      
    @property
    def framework(self):
      # TODO(lynn): checkme
      for file in self.list_files():
        # hf-transformers/diffusers可以通过这种方式判断
        if file.endswith(".bin") or file.endswith(".safetensors"):
          return "pt"
      # 由于mindformers的模型权重文件后缀为.ckpt过于通用，因此默认不是pt就是ms
      return "ms"
    
    @property
    def backend(self):
      # backend：transformers/diffusers/mindformers
      # TODO(lynn): checkme
      if self.framework == "pt":
        for file in self.list_files():
          # 需要确认下这个调试是否过于严苛
          if file.endswith(".json"):
            with open(file, "r", encoding="utf-8") as fp:
              config = json.load(fp)
            if "_diffusers_version" in config:
              return "diffusers"
            elif "transformers_version" in config:
              return "transformers"
      else:
        # TODO
        return ""
    
    @property
    def model_type(self):
      if self.backend == "transformers":
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as fp:
           config = json.load(fp)
           return config.get("model_type", "unknown_model_type")
      elif self.backend == "diffusers":
        # TODO(lynn): checkme
        config_path = os.path.join(self.model_dir, "model_index.json")
        with open(config_path, "r", encoding="utf-8") as fp:
           config = json.load(fp)
           return config.get("_class_name", "unknown_model_type")
      # TODO: mindspore?
      return "unknown_model_type"
    
    @property
    def pipeline_task(self):
      """
      解析Markdown文件中的YAML元数据
      """
      if self.is_local:
        readme_path = os.path.join(self.model_dir, "README.md")
        with open(readme_path, 'r', encoding='utf-8') as fp:
           content = fp.read()
        yaml_pattern = re.compile(r'^---.*?---$', re.DOTALL | re.MULTILINE)
        match = yaml_pattern.search(content)
        if match:
          yaml_content = match.group(0)
          # 去除---前后的空行
          yaml_content = re.sub(r'^\s*---\s*$', '', yaml_content, flags=re.MULTILINE).strip()
          try:
            # 使用yaml库解析YAML内容
            import yaml
            metadata = yaml.safe_load(yaml_content)
            return metadata.get("pipeline_tag")
          except ImportError:
            raise ImportError("yaml库未安装，请使用pip install PyYAML来安装")
        return {}
    
    @property
    def metadata(self):
      return {
        "model_dir": self.model_dir,
        "is_local": self.is_local,
        "model_type": self.model_type,
        "pipeline_tag": self.pipeline_task,
        "framework": self.framework,
        "backend": self.backend,
      }


def _can_load_by_hf_automodel(automodel_class: type, config) -> bool:
    automodel_class_name = automodel_class.__name__
    if type(config) in automodel_class._model_mapping.keys():
        return True
    if hasattr(config, 'auto_map') and automodel_class_name in config.auto_map:
        return True
    return False


def get_hf_automodel_class(model_dir: str,
                           task_name: Optional[str]) -> Optional[type]:
    from transformers import (AutoConfig,
                              AutoModel,
                              AutoModelForCausalLM,
                              AutoModelForSeq2SeqLM,
                              AutoModelForTokenClassification,
                              AutoModelForSequenceClassification)
    automodel_mapping = {
        "chat": AutoModelForCausalLM,
    }
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        return None
    try:
        # FIXME
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # if task_name is None:
        #     automodel_class = get_default_automodel(config)
        # else:
        #     automodel_class = automodel_mapping.get(task_name, None)
        automodel_class = automodel_mapping.get(task_name, None)

        if automodel_class is None:
            return None
        if _can_load_by_hf_automodel(automodel_class, config):
            return automodel_class
        if (automodel_class is AutoModelForCausalLM
                and _can_load_by_hf_automodel(AutoModelForSeq2SeqLM, config)):
            return AutoModelForSeq2SeqLM
        return None
    except Exception:
        return None


def try_to_load_hf_model(model_dir: str, task_name: str, **kwargs):
    automodel_class = get_hf_automodel_class(model_dir, task_name)

    model = None
    if automodel_class is not None:
        # use hf
        model = automodel_class.from_pretrained(model_dir, **kwargs)
    return model