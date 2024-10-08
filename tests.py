import re
import os
import json


def parse_yaml_metadata(model_dir):
    """
    解析Markdown文件中的YAML元数据
    :param markdown_text: Markdown文件的内容
    :return: 元数据字典
    """
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, 'r', encoding='utf-8') as file:
        # 读取文件内容
        markdown_content = file.read()
    breakpoint()
    yaml_pattern = re.compile(r'^---.*?---$', re.DOTALL | re.MULTILINE)
    match = yaml_pattern.search(markdown_content)
    if match:
        breakpoint()
        yaml_content = match.group(0)
        # 去除---前后的空行
        yaml_content = re.sub(r'^\s*---\s*$', '', yaml_content, flags=re.MULTILINE).strip()
        breakpoint()
        try:
            # 使用yaml库解析YAML内容
            import yaml
            metadata = yaml.safe_load(yaml_content)
            return metadata
        except ImportError:
            raise ImportError("yaml库未安装，请使用pip install PyYAML来安装")
    return {}

# # 示例Markdown文本


metadata = parse_yaml_metadata("/home/lynn/github/qwen2.5-0.5b-instruct")
print(metadata.get("pipeline_tag"))


# with open("/home/lynn/github/qwen2.5-0.5b-instruct/README.md", 'r', encoding='utf-8') as file:
#     # 读取文件内容
#     markdown_content = file.read()

# # 打印文件内容
# print(markdown_content)