from pipelines import pipeline


if __name__ == "__main__":
  pipe = pipeline(
    task="text-classification",
    model="/home/lynn/github/distilbert-base-uncsed",
    framework="pt",
    backend="transformers")
  print(pipe("This movie is disgustingly good!"))

  pipe = pipeline(
    task="text-to-image",
    model="/home/lynn/github/sd1-4",
    framework="pt",
    backend="diffusers")
  res = pipe("a photo of an astronaut riding a horse on mars")
  res.save("astronaut_rides_horse.png")
