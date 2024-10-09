

def hf_native_pipeline():
  from transformers import pipeline

  pipe = pipeline(task="text-classification", model="/home/lynn/github/distilbert-base-uncsed")
  print(pipe("This movie is disgustingly good!"))


def tr_wrapper_pipeline():
  from hf.hf import HFPipelineWrapper
  pipe = HFPipelineWrapper.create_from_model_str("/home/lynn/github/distilbert-base-uncsed")
  print(pipe("This movie is disgustingly good!"))


def di_wrapper_pipeline():
  from hf.hf import HFPipelineWrapper
  pipe = HFPipelineWrapper.create_from_model_str("/home/lynn/github/sd1-4")
  res = pipe("a photo of an astronaut riding a horse on mars")
  res.save("astronaut_rides_horse.png")


if __name__ == "__main__":
  # hf_native_pipeline()
  di_wrapper_pipeline()
