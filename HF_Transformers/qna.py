from transformers import pipeline

qna = pipeline(
    "question-answering",
    model = "monologg/koelectra-base-v3-finetuned-korquad",
    tokenizer = "monologg/koelectra-base-v3-finetuned-korquad"
)

context = (
    "서울은 대한민국의 수도이자 특별시입니다."
    "한반도 중앙에 위치한 서울은 정치, 경제, 사회, 문화의 중심지입니다."
)

question = "서울은 어디에 위치해 있나요?"

result = qna(question=question, context=context)

print(result)