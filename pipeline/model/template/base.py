import enum
from abc import ABC
from typing import Sequence, Dict


class TemplateType(enum.Enum):
    LLM = enum.auto(),
    MLLM = enum.auto()


# template的基本信息可以放在这里
# template的概念主要来自大模型，会对输入的prompt和机器的回复，按照单轮和多轮进行特定格式的encode，以符合模型要求
class BaseTemplate(ABC):
    def __init__(self, template_type: TemplateType):
        self.template_type = template_type

    def encode(self, **kwargs):
        raise NotImplementedError
    

class BaseLlmTemplate(BaseTemplate):
    def __init__(self,
                 prefix: str = None):
        super().__init__(TemplateType.LLM)
        self.prefix = prefix

    def encode(self, messages: Sequence[Dict[str, str]], **kwargs):
        # 举个例子
        for message in messages:
            if message["role"] == "user":
                message["content"] = self.prefix + " " + message["content"]
        return messages