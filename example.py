from zeroshot import pipelines
from transformers import pipeline

model = "dbmdz/bert-base-turkish-cased"
nsp_zs = pipeline(task="nsp-zeroshot-classification", model=model)

text = "Bu yıl futbol çok ilerledi, çok gol atıldı"
outputs = nsp_zs(
    text,
    hypothesis_template="Bu cümle {} ile ilgilidir.",
    candidate_labels=["spor", "müzik", "siyaset"],
    reverse=False,
)

print("Input:", text)
print("Outputs:", outputs)
