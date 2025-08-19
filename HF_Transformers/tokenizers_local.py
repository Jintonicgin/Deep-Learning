from transformers import AutoModel, AutoTokenizer

# 모델 이름 지정
model_name = "bert-base-uncased"

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 입력 텍스트
text = "i am learning about tokenizers"

# 텍스트를 토큰화하고 텐서로 변환
inputs = tokenizer(text, return_tensors="pt")

# 모델 추론
outputs = model(**inputs)

# 마지막 히든 스테이트 텐서와 shape 출력
print(outputs.last_hidden_state.shape)