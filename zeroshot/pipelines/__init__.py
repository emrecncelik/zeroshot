from .nsp_zeroshot_classification import NSPZeroShotClassificationPipeline

from transformers.pipelines import PIPELINE_REGISTRY
from transformers import BertForNextSentencePrediction, TFBertForNextSentencePrediction

PIPELINES = [
    dict(
        task="nsp-zeroshot-classification",
        pipeline_class=NSPZeroShotClassificationPipeline,
        pt_model=BertForNextSentencePrediction,
        tf_model=TFBertForNextSentencePrediction,
        default={"pt": ("bert-base-uncased")},
        type="text",
    )
]

for pipeline in PIPELINES:
    PIPELINE_REGISTRY.register_pipeline(**pipeline)
