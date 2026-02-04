from src.lab.sentiment import SentimentRequest, classify_sentiment, build_default_context

ctx = build_default_context()
req = SentimentRequest(text="I love the design, but the setup was frustrating.")
result = classify_sentiment(req, ctx)
print(result)
