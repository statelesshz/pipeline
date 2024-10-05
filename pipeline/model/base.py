from typing import List, Optional, Union


class BaseModel:
  def __init__(self,
                tasks: Union[List[str], str],
                model_id_or_path: Optional[str],
                requires: Optional[List[str]]):
    if type(tasks) is str:
      tasks = [tasks]
    self.tasks = tasks
    self.model_id_or_path = model_id_or_path
    self.requires = requires
    self.framework = None
  
  def __call__(self, *args, **kwargs):
    raise NotImplementedError
  
  def forward(self, *args, **kwargs):
    raise NotImplementedError
  
  def init_and_load(self, framework):
    self.check_requirements()
    self.framework = framework

  def check_requirements(self):
    print("start check model requirements")
    if self.requires:
      for require in self.requires:
        # target:transformers, cann
        # target_type: python, ascend
        # version
        print(require)


# class BaseLlmModel(BaseModel):
#     def __init__(self,
#                  task_name_list: Union[List[str], str],
#                  model_group_name: str,
#                  model_id_or_path: str,
#                  requires: Optional[List[str]]):
#         super().__init__(task_name_list, model_group_name, model_id_or_path, requires)

#     def is_chat_model(self):
#         return "instruct" in self.model_id_or_path or "chat" in self.model_id_or_path

#     def init_and_load(self, framework):
#         # TODO from transformers/mindformers import AutoModel, ...
#         super().init_and_load()


# register model, such as: qwen2-0.5b-instruct
