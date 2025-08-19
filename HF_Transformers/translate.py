from huggingface_hub import login
from transformers import pipeline
from dotenv import load_dotenv
import os

# Hugging Face 로그인 (토큰 입력)
load_dotenv()
login(token=os.getenv("LOGIN_TOKEN"))

# Facebook NLLB 모델 사용 (3.3B 모델: 높은 성능, 600M 모델: 속도 빠름)
model_name = "facebook/nllb-200-distilled-600M" # 또는 "facebook/nllb-200-3.3B"

# 한국어 -> 영어 번역
translator_ko_en = pipeline("translation", model=model_name, src_lang="kor_Hang", tgt_lang="eng_Latn")

# 영어 -> 한국어 번역
translator_en_ko = pipeline("translation", model=model_name, src_lang="eng_Latn", tgt_lang="kor_Hang")

# 테스트 문장
korean_text = "안녕하세요. 오늘 날씨는 어떤가요?"
english_translation = translator_ko_en(korean_text)
print("Korean to English:", english_translation[0]['translation_text'])

english_text = "Hello, how is the weather today?"
korean_translation = translator_en_ko(english_text)
print("English to Korean", korean_translation[0]['translation_text'])
