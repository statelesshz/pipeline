from pipelines import pipeline

def text_generation_with_only_task():
  pipe = pipeline(task="text-generation", device="npu:0")
  res = pipe(
    inputs=[{
      "role": "user",
      "content": "Give me a short introduction to large language model."
    }],
    max_new_tokens=512,
    do_sample=True
  )
  print(res)

def text_generation_with_task_model_framework_backend():
  pipe = pipeline(
    task="text-generation",
    model="/home/lynn/github/qwen2.5-0.5b-instruct",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    inputs=[{
      "role": "user",
      "content": "Give me a short introduction to large language model."
    }],
    max_new_tokens=512,
    do_sample=True
  )
  print(res)



if __name__ == "__main__":
  # text_generation_with_only_task()
  text_generation_with_task_model_framework_backend()
