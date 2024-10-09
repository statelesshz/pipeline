# from pipeline.api import parse_model_metadata
from pipeline.utils import ModelInfo
from pipeline import pipeline


if __name__ == "__main__":
  model_str = "/home/lynn/github/qwen2.5-0.5b-instruct"
  model_info = ModelInfo(model_str)
  model_type = model_info.model_type
  pipe = pipeline(task="chat", model=model_str)
  breakpoint()
