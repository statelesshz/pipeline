from transformers import pipeline

pipe = pipeline(task="text-classification", model="/home/lynn/github/distilbert-base-uncsed")
print(pipe("This movie is disgustingly good!"))
