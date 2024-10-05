
from abc import ABC
from typing import Dict, List, Optional, Sequence

from .framework import FrameWork
from .utils import try_to_load_hf_model


class BasePipelineWrapper(ABC):
  def __init__(self,
               task: str,
               default_framework: FrameWork = None,
               default_model: str = None,
               **kwargs):
     self.task = task
     self.default_framework = default_framework
     self.default_model = default_model

     self.pipeline = None
     self.framework = None

  def set_pipeline(self, pipeline):
     self.pipeline = pipeline
     self.framework = pipeline.framework

  def __call__(self, *args, **kwargs):
     raise NotImplementedError
  
  def _preprocess(self, *args, **kwargs):
     raise NotImplementedError

  def _forward(self, **kwargs):
      raise NotImplementedError

  def _postprocess(self, **kwargs):
      raise NotImplementedError


class BasePipeline(ABC):
   def __init__(self,
                task: str,
                requirements: Optional[List[str]],
                **kwargs):
      self.task = task
      if type(requirements) is str:
         requirements = [requirements]
      self.requirements = requirements
      # framework&model需要真正执行时通过init_and_load来确定
      self.framework = None
      self.model_str = None
      self.model = None

   def init_and_load(self, model_str, *args, **kwargs):
      if self.requirements:
         print("check requirement:")
         for requirement in self.requirements:
            print(requirement)
   
   def preprocess(self, **kwargs):
      raise NotImplementedError
   
   def forward(self, **kwargs):
      raise NotImplementedError
   
   def postprocess(self, **kwargs):
      raise NotImplementedError
   

class BasePtPipeline(BasePipeline):
   def __init__(self,
                task: str,
                requirements: Optional[List[str]] = None,
                **kwargs):
      super().__init__(task, requirements, **kwargs)
      self.framework = FrameWork.PT
   
   def init_and_load(self, model_str, *args, **kwargs):
      super().init_and_load(model_str, *args, **kwargs)
      # get openmind_model
      """
      model = get_openmind_model_by_id(model_str)
      model.init_and_load()
      self.model = model
      """


class BaseMsPipeline(BasePipeline):
   def __init__(self,
                task: str,
                requirements: Optional[List[str]] = None,
                **kwargs):
      super().__init__(task, requirements, **kwargs)
      self.framework = FrameWork.MS
   
   def init_and_load(self, model_str, **kwargs):
      super().init_and_load(model_str, **kwargs)


# register pipeline_wrapper & pipeline
class ChatPipelineWrapper(BasePipelineWrapper):
   def __init__(self,
                default_framework: FrameWork = None,
                default_model: str = None,
                **kwargs):
      super().__init__("chat", default_framework, default_model, **kwargs)
   
   def __call__(self, msg: Sequence[Dict[str, str]], **kwargs):
      msg = self._preprocess(msg, **kwargs)
      resp = self._forward(msg, **kwargs)
      resp = self._postprocess(resp, **kwargs)
      return resp
   
   def _preprocess(self,  messages: Sequence[Dict[str, str]], **kwargs):
      return self.pipeline.preprocess(messages, **kwargs)
   
   def _forward(self, messages: Sequence[Dict[str, str]], **kwargs):
        return self.pipeline.forward(messages, **kwargs)

   def _postprocess(self, response, **kwargs):
        return self.pipeline.postprocess(response, **kwargs)


class ChatPtPipeline(BasePtPipeline):
    def __init__(self,
                 requirements: Optional[List[str]],
                 **kwargs):
        super().__init__("chat", requirements, **kwargs)

    def init_and_load(self, model_str, *args, **kwargs):
       super().init_and_load(model_str, *args, **kwargs)
       # load model from transformers
       model = try_to_load_hf_model(model_dir=model_str, task_name="chat", **kwargs)
       self.model = model


    def preprocess(self, messages: Sequence[Dict[str, str]], **kwargs):
        return self.template.encode(messages, **kwargs)

    def forward(self, messages: Sequence[Dict[str, str]], **kwargs):
        return {"role": "assistant", "content": "hello"}

    def postprocess(self, response, **kwargs):
        response["content"] = response["content"] + " postprocess"
        return response


PIPELINE_WRAPPER_MAPPING: Dict[str, BasePipelineWrapper] = {}
PIPELINE_MAPPING: Dict[str, Dict[FrameWork, BasePipeline]] = {}


def register_pipeline_wrapper(task: str,
                              pipeline_wrapper: BasePipelineWrapper,
                              **kwargs):
   pipeline_wrapper.task = task
   PIPELINE_WRAPPER_MAPPING[task] = pipeline_wrapper


def register_pipeline(task: str,
                      framework: FrameWork,
                      pipeline: BasePipeline,
                      **kwargs):
   if task not in PIPELINE_MAPPING:
      PIPELINE_MAPPING[task] = {}
   pipeline.task = task
   PIPELINE_MAPPING[task][framework] = pipeline

# 注册pipeline wrapper
register_pipeline_wrapper("chat", ChatPipelineWrapper(default_framework=FrameWork.PT, default_model="/home/lynn/github/qwen2.5-0.5b-instruct"))

# 注册pipeline
register_pipeline("chat", FrameWork.PT, ChatPtPipeline(requirements="transformers>4.42"))
