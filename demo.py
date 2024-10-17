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

def text_generation_with_task_model_framework_pt():
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

def visual_question_answering_with_task_model_framework_pt():
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

def zero_shot_object_detection_with_task_model_framework_pt():
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

def zero_shot_classification_with_task_model_framework_pt():
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

def depth_estimation_with_task_model_framework_pt():
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

def image_to_image_with_task_model_framework_pt():
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

def mask_generation_with_task_model_framework_pt():
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

def fill_mask_with_only_task():
  pipe = pipeline(task="fill-mask", device="npu:0")
  res = pipe(
    inputs="This is a simple [MASK]."
  )
  print(res)

def fill_mask_with_task_model_framework_pt():
  pipe = pipeline(
    task="fill-mask",
    model="PyTorch-NPU/bert_base_uncased",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    inputs="This is a simple [MASK]."
  )
  print(res)

def question_answering_with_only_task():
  pipe = pipeline(task="question-answering", device="npu:0")
  res = pipe(
    question="Where do I live?",
    context="My name is Wolfgang and I live in Berlin."
  )
  print(res)

def question_answering_with_task_model_framework_pt():
  pipe = pipeline(
    task="question-answering",
    model="PyTorch-NPU/roberta_base_squad2",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    question="Where do I live?",
    context="My name is Wolfgang and I live in Berlin."
  )
  print(res)

def summarization_with_oly_task():
  pipe = pipeline(task="summarization", device="npu:0")
  res = pipe(
    inputs="An apple a day, keeps the doctor away",
    min_length=5,
    max_length=20,
  )
  print(res)

def summarization_with_task_model_framework_pt():
  pipe = pipeline(
    task="summarization",
    model="PyTorch-NPU/bart_large_cnn",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    inputs="An apple a day, keeps the doctor away",
    min_length=5,
    max_length=20,
  )
  print(res)

def table_question_answering_with_only_task():
  pipe = pipeline(task="table-question-answering", device="npu:0")
  res = pipe(
    query="How many stars does the transformers repository have?",
    table={
        "Repository": ["Transformers", "Datasets", "Tokenizers"],
        "Stars": ["36542", "4512", "3934"],
        "Contributors": ["651", "77", "34"],
        "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
    },
  )
  print(res)

def table_question_answering_with_task_model_framework_pt():
  pipe = pipeline(
    task="table-question-answering",
    model="PyTorch-NPU/tapas_base_finetuned_wtq",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    query="How many stars does the transformers repository have?",
    table={
        "Repository": ["Transformers", "Datasets", "Tokenizers"],
        "Stars": ["36542", "4512", "3934"],
        "Contributors": ["651", "77", "34"],
        "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
    },
  )
  print(res)

def translation_with_only_task():
  pipe = pipeline(task="translation", device="npu:0")
  res = pipe(
    "How old are you?"
  )
  print(res)

def translation_with_task_model_framework_pt():
  pipe = pipeline(
    task="translation",
    model="PyTorch-NPU/t5_base",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    "How old are you?"
  )
  print(res)

def text_classification_with_only_task():
  pipe = pipeline(task="text-classification", device="npu:0")
  res = pipe(
    "This movie is disgustingly good !"
  )
  print(res)

def text_classification_with_task_model_framework_pt():
  pipe = pipeline(
    task="text-classification",
    model="PyTorch-NPU/distilbert_base_uncased_finetuned_sst_2_english",
    framework="pt",
    backend="transformers",
    device="npu:0"
  )
  res = pipe(
    "This movie is disgustingly good !"
  )
  print(res)

def text_classification_with_task_model_framework_ms():
  pipe = pipeline(
    task="text-classification",
    model="/home2/cqq/models_save/glm2_6b_ms",
    framework="pt",
    backend="mindformers",
    device_id=0
  )
  res = pipe(
    "Give me some advice on how to stay healthy."
  )
  print(res)

if __name__ == "__main__":
  # text_generation_with_only_task()
  # text_generation_with_only_model()
  # text_generation_with_task_model_framework_pt()
  # visual_question_answering_with_task_model_framework_pt()
  # zero_shot_object_detection_with_task_model_framework_pt()
  # zero_shot_classification_with_task_model_framework_pt()
  # depth_estimation_with_task_model_framework_pt()
  # image_to_image_with_task_model_framework_pt()
  # mask_generation_with_task_model_framework_pt()
  # zero_shot_image_classification_with_only_task()
  # feature_extraction_with_only_task()
  # image_classification_with_only_task()
  # image_to_text_with_only_task()
  # text2text_generation_with_only_task()
  # token_classification_with_only_task()
  # fill_mask_with_only_task()
  # fill_mask_with_task_model_framework_pt()
  # question_answering_with_only_task()
  # question_answering_with_task_model_framework_pt()
  # summarization_with_oly_task()
  # summarization_with_task_model_framework_pt()
  # table_question_answering_with_only_task()
  # table_question_answering_with_task_model_framework_pt()
  # translation_with_only_task()
  # translation_with_task_model_framework_pt()
  # text_classification_with_only_task()
  # text_classification_with_task_model_framework_pt()
  text_classification_with_task_model_framework_ms()
