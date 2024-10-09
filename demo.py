from pipelines import pipeline


if __name__ == "__main__":
  pipe = pipeline(
    task="text-classification",
    model="/home/lynn/github/distilbert-base-uncsed",
    framework="pt",
    backend="transformers")
  print(pipe("This movie is disgustingly good!"))

  print("\n\n")

  pipe = pipeline(
    task="text-to-image",
    model="/home/lynn/github/sd1-4",
    framework="pt",
    backend="diffusers")
  res = pipe("a photo of an astronaut riding a horse on mars")
  res.save("astronaut_rides_horse.png")


  print("\n\n")

  pipe = pipeline(
    task = "text-generation",
    model="/home/lynn/github/qwen2.5-0.5b-instruct",
    framework="pt",
    backend="transformers")
  
  prompt = "Give me a short introduction to large language model."
  input = [{"role": "user", "content": prompt}]
  res = pipe(input, max_new_tokens=512, do_sample=True)
  print(res)
