from transformers import pipeline

sentiment_analyzer_ko = pipeline("text-classification", model = "beomi/KcELECTRA-base-v2022")

korean_text = ["이 제품 정말 마음에 들어요!", "완전 별로네요. 다시는 안 살 겁니다.", "그냥 보통이에에요."]

results_ko = sentiment_analyzer_ko(korean_text)

for text, result in zip(korean_text, results_ko):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}")
    print("-" * 40)