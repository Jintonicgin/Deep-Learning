from transformers import pipeline

ner = pipeline(
    "ner",
    model="Davlan/bert-base-multilingual-cased-ner-hrl",
    tokenizer="Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple"
)

text = "삼성전자는 수원에 본사를 두고 있으며, 이재용 회장이 경영을 맡고 있다."
results = ner(text)

for entity in results:
    print(f"{entity['word']} ({entity['entity_group']}) - score: {entity['score']:.4f}")
