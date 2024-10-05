from pipelines import pipeline


if __name__ == "__main__":
  pipe = pipeline(
    task="text-generation",
    model="/home/lynn/github/qwen2.5-0.5b-instruct",
    framework="pt",
    backend="transformers",
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
