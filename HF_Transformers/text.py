from transformers import pipeline

# KoGPT 모델을 사용한 텍스트 생성 파이프라인
text_generator = pipeline("text-generation", model="skt/kogpt2-base-v2")

# 입력 문장
prompt = "오늘 날씨가 참 좋네요"

# 텍스트 생성 실행
result = text_generator(prompt, max_length=50, do_sample=True, # 샘플링을 사용할지 여부(True: 랜덤성 증가, False: 항상 동일한 출력)
                        top_k=50, # 확률이 높은 50개의 단어만 고려함
                        top_p=0.95) # 확률의 누적합이 0.95 이하인 단어들만 선택함

print(result[0]['generated_text'])