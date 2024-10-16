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

def text_generation_with_only_model():
  pipe = pipeline(model="AI_Connect/Qwen2_0.5B", device="npu:0")
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

def visual_question_answering_with_task_model_framework_backend():
  pipe = pipeline(
    task="visual-question-answering",
    model="/home2/cqq/models_save/blip-vqa-base",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    image="./fixtures/tests_samples/COCO/000000039769.png",
    question="How many cats are there?",
    top_k=2,
    do_sample=True
  )
  print(res)

def zero_shot_object_detection_with_task_model_framework_backend():
  pipe = pipeline(
    task="zero-shot-object-detection",
    model="/home2/cqq/models_save/owlvit-base-patch32",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    image="./fixtures/tests_samples/COCO/000000039769.png",
    candidate_labels=["cat", "remote", "couch"],
    do_sample=True
  )
  print(res)

def zero_shot_classification_with_task_model_framework_backend():
  pipe = pipeline(
    task="zero-shot-classification",
    model="/home2/cqq/models_save/deberta-v3-large-zeroshot-v2.0",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    sequences="Who are you voting in 2020",
    candidate_labels=["politics", "public health", "science"],
  )
  print(res)

def depth_estimation_with_task_model_framework_backend():
  pipe = pipeline(
    task="depth-estimation",
    model="/home2/cqq/models_save/dpt-large",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    images="./fixtures/tests_samples/COCO/000000039769.png",
  )
  print(res)

def image_to_image_with_task_model_framework_backend():
  pipe = pipeline(
    task="image-to-image",
    model="/home2/cqq/models_save/swin2SR-classical-sr-x2-64",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    images="./fixtures/tests_samples/COCO/000000039769.png",
  )
  print(res)

def mask_generation_with_task_model_framework_backend():
  pipe = pipeline(
    task="mask-generation",
    model="/home2/cqq/models_save/sam-vit-base",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    image="./fixtures/tests_samples/COCO/000000039769.png",
    points_per_batch=256,
  )
  print(res)

def zero_shot_image_classification_with_only_task():
  pipe = pipeline(task="zero-shot-image-classification", device="npu:0")
  res = pipe(
    "./fixtures/tests_samples/COCO/000000039769.png",
    candidate_labels=["animals", "humans", "landscape"],
  )
  print(res)

def feature_extraction_with_only_task():
  pipe = pipeline(task="feature-extraction", device="npu:0")
  res = pipe("This is a simple test.", return_tensors=True)
  print(res.shape)

def image_classification_with_only_task():
  pipe = pipeline(task="image-classification", device="npu:0")
  res = pipe("./fixtures/tests_samples/COCO/000000039769.png")
  print(res)

def image_to_text_with_only_task():
  pipe = pipeline(task="image-to-text", device="npu:0")
  res = pipe("./fixtures/tests_samples/COCO/000000039769.png")
  print(res)

def text2text_generation_with_only_task():
  pipe = pipeline(task="text2text-generation", device="npu:0")
  res = pipe("translate English to German: How old are you?")
  print(res)

def token_classification_with_only_task():
  pipe = pipeline(task="token-classification", device="npu:0")
  res = pipe("Apple est créée le 1er avril 1976 dans le garage de la maison d'enfance de Steve Jobs à Los Altos en Californie par Steve Jobs, Steve Wozniak et Ronald Wayne14, puis constituée sous forme de société le 3 janvier 1977 à l'origine sous le nom d'Apple Computer, mais pour ses 30 ans et pour refléter la diversification de ses produits, le mot « computer » est retiré le 9 janvier 2015.")
  print(res)

if __name__ == "__main__":
  # text_generation_with_only_task()
  # text_generation_with_only_model()
  # text_generation_with_task_model_framework_backend()
  # visual_question_answering_with_task_model_framework_backend()
  # zero_shot_object_detection_with_task_model_framework_backend()
  # zero_shot_classification_with_task_model_framework_backend()
  # depth_estimation_with_task_model_framework_backend()
  # image_to_image_with_task_model_framework_backend()
  # mask_generation_with_task_model_framework_backend()
  # zero_shot_image_classification_with_only_task()
  # feature_extraction_with_only_task()
  # image_classification_with_only_task()
  # image_to_text_with_only_task()
  # text2text_generation_with_only_task()
  # token_classification_with_only_task()