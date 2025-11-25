import re
from collections import Counter

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very'
    ])
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(filtered_words)

def tokenize(text):
    return text.split()

def analyze_sentiment(text):
    positive_words = set([
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'positive', 'love', 'happy', 'joy', 'success', 'best', 'perfect',
        'beautiful', 'brilliant', 'awesome', 'outstanding', 'satisfied'
    ])
    negative_words = set([
        'bad', 'terrible', 'horrible', 'awful', 'poor', 'negative',
        'hate', 'sad', 'failure', 'worst', 'ugly', 'disappointing',
        'problem', 'issue', 'difficult', 'hard', 'pain'
    ])
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    total = positive_count + negative_count
    if total == 0:
        return 'Neutral', 0.5
    sentiment_score = positive_count / total
    if sentiment_score > 0.6:
        return 'Positive', sentiment_score
    elif sentiment_score < 0.4:
        return 'Negative', sentiment_score
    else:
        return 'Neutral', sentiment_score

def extract_topics(text, num_topics=3):
    words = text.split()
    word_freq = Counter(words)
    topics = []
    for word, freq in word_freq.most_common(num_topics * 3):
        if len(word) > 4:
            topics.append({'word': word, 'frequency': freq})
        if len(topics) == num_topics:
            break
    return topics

def extractive_summarization(text, num_sentences=2):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) <= num_sentences:
        return sentences
    words = tokenize(clean_text(text))
    word_freq = Counter(words)
    sentence_scores = []
    for sentence in sentences:
        score = sum(word_freq.get(word, 0) for word in tokenize(clean_text(sentence)))
        sentence_scores.append((sentence, score))
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in sentence_scores[:num_sentences]]

def generate_insights(sentiment, topics, word_count):
    insights = []
    recommendations = []
    if sentiment == 'Positive':
        insights.append("Overall positive sentiment detected in the text.")
        recommendations.append("Leverage this positive tone in communications.")
    elif sentiment == 'Negative':
        insights.append("Negative sentiment detected in the text.")
        recommendations.append("Address negative aspects and investigate root causes.")
    else:
        insights.append("Neutral sentiment detected in the text.")
        recommendations.append("Monitor for changes in future data.")
    if topics:
        top_topic = topics[0]['word']
        insights.append(f"Main focus area: '{top_topic}' appears most frequently.")
        recommendations.append(f"Explore '{top_topic}' for more details.")
    if word_count < 50:
        insights.append("Brief text with limited content.")
    elif word_count > 500:
        insights.append("Comprehensive text with rich details.")
    return insights, recommendations
