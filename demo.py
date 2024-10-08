# from pipeline.api import parse_model_metadata
from pipeline.utils import ModelInfo
from pipeline import pipeline


if __name__ == "__main__":
  model_str = "/home/lynn/github/qwen2.5-0.5b-instruct"
  # config_files = _find_config_json(model_str)
  # print(config_files)
  # config = read_config_json(model_str)
  # print(config["model_type"])
  # task = parse_model_metadata(model_str=model_str)
  # print(task)
  model_info = ModelInfo(model_str)
  model_type = model_info.model_type
  pipe = pipeline(task="chat", model=model_str)
  breakpoint()
