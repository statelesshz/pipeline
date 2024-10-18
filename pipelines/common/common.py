from ..base import BasePipelineWrapper

class TextGenerationPipelineWrapper(BasePipelineWrapper):
  task: str = "text-generation"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "Baichuan/Baichuan2_7b_chat_pt@ca161b7"


class VisualQuestionAnsweringPipelineWrapper(BasePipelineWrapper):
  task: str = "visual-question-answering"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/blip_vqa_base@4450392"


class ZeroShotClassificationPipelineWrapper(BasePipelineWrapper):
  task: str = "zero-shot-classification"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/deberta_v3_large_zeroshot_v2.0@d38d6f4"


class DepthEstimationPipelineWrapper(BasePipelineWrapper):
  task: str = "depth-estimation"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/dpt_large@270fa97"


class ImageToImagePipelineWrapper(BasePipelineWrapper):
  task: str = "image-to-image"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/swin2SR_classical_sr_x2_64@407e816"


class MaskGenerationPipelineWrapper(BasePipelineWrapper):
  task: str = "mask-generation"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/sam_vit_base@d0ad399"


class ZeroShotImageClassificationPipelineWrapper(BasePipelineWrapper):
  task: str = "zero-shot-image-classification"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/siglip_so400m_patch14_384@b4099dd"


class FeatureExtractionPipelineWrapper(BasePipelineWrapper):
  task: str = "feature-extraction"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/xlnet_base_cased@bc7408f"


class ImageClassificationPipelineWrapper(BasePipelineWrapper):
  task: str = "image-classification"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/beit_base_patch16_224@a46c2b5"


class ImageToTextPipelineWrapper(BasePipelineWrapper):
  task: str = "image-to-text"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/blip-image-captioning-large@059b23b"


class Text2TextGenerationPipelineWrapper(BasePipelineWrapper):
  task: str = "text2text-generation"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/flan_t5_base@d15ab63"


class TokenClassificationPipelineWrapper(BasePipelineWrapper):
  task: str = "token-classification"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/camembert_ner@1390d33"


class FillMaskPipelineWrapper(BasePipelineWrapper):
  task: str = "fill-mask"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/bert_base_uncased@42ad83b"


class QuestionAnsweringPipelineWrapper(BasePipelineWrapper):
  task: str = "question-answering"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/roberta_base_squad2@ba973aa"


class SummarizationPipelineWrapper(BasePipelineWrapper):
  task: str = "summarization"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/bart_large_cnn@b39bb57"


class TableQuestionAnsweringPipelineWrapper(BasePipelineWrapper):
  task: str = "table-question-answering"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/tapas_base_finetuned_wtq@17e5ded"


class TranslationPipelineWrapper(BasePipelineWrapper):
  task: str = "translation"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/t5_base@68829a3"


class TextClassificationPipelineWrapper(BasePipelineWrapper):
  task: str = "text-classification"
  framework: str = "pt"
  backend: str = "transformers"
  model_id: str = "PyTorch-NPU/distilbert_base_uncased_finetuned_sst_2_english@5a5cb27"
