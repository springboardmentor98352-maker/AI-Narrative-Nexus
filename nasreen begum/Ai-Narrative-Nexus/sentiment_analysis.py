from textblob import TextBlob

def analyze_sentiment(text: str):

    if not text.strip():
        return "No text", 0.0

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive 😊"
    elif polarity < 0:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    return sentiment, polarity
    

