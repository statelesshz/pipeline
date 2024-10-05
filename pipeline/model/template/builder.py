from typing import Dict

from .base import BaseTemplate, BaseLlmTemplate

TEMPLATE_MAPPING: Dict[str, BaseTemplate] = {}


# 跟 TemplateName 和 ModelGroupName 是 1对多的关系， 多个group 可能共用一个 template
class TemplateName:
    QWEN2 = "qwen2"


def register_template(template_name: str, template: BaseTemplate, **kwargs):
    TEMPLATE_MAPPING[template_name] = template

register_template(TemplateName.QWEN2, BaseLlmTemplate("qwen2 prefix"))
