import os
import nltk # type: ignore
from nltk.sentiment import SentimentIntensityAnalyzer # type: ignore
from collections import Counter
import matplotlib.pyplot as plt # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.decomposition import LatentDirichletAllocation # type: ignore
import numpy as np # type: ignore

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()

# ------------ BASIC METRICS ---------------- #


def word_count(text):
    return len(text.split())


def sentence_count(text):
    return len([s for s in text.split(".") if s.strip()])


def sentiment_analysis(text):
    return sia.polarity_scores(text)


def sentiment_distribution(sentiment_scores):
    return {
        "Positive": sentiment_scores.get("pos", 0),
        "Negative": sentiment_scores.get("neg", 0),
        "Neutral": sentiment_scores.get("neu", 0)
    }


def sentiment_to_emoji(score):
    if score > 0.2:
        return " Positive"
    elif score < -0.2:
        return " Negative "
    else:
        return " Neutral"


def sentiment_distribution_chart(distribution):
    labels = list(distribution.keys())
    values = list(distribution.values())

    fig, ax = plt.subplots()
    ax.bar(labels, values)  # default colors
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment Type")
    ax.set_ylabel("Score")
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    return fig


def top_tokens(text, n=10):
    tokens = text.split()
    return Counter(tokens).most_common(n)


def simple_summary(text):
    #extractive summary.
    sentences = text.split(".")
    if len(sentences) > 2:
        return sentences[0].strip() + ". " + sentences[-2].strip() + "."
    else:
        return text[:1000] + "..."


def comprehensive_summary(text, sentiment_scores, tokens):
    """Generate a comprehensive paragraph summary of the text analysis"""
    wc = word_count(text)
    sc = sentence_count(text)
    compound = sentiment_scores.get("compound", 0)
    
    # Sentiment interpretation
    if compound > 0.2:
        sentiment_desc = "predominantly positive tone"
    elif compound < -0.2:
        sentiment_desc = "predominantly negative tone"
    else:
        sentiment_desc = "neutral sentiment"
    
    # Word length interpretation
    avg_word_length = len(text) / max(wc, 1)
    if avg_word_length > 6:
        complexity = "sophisticated and complex language"
    elif avg_word_length > 4.5:
        complexity = "moderate vocabulary complexity"
    else:
        complexity = "simple and accessible language"
    
    # Key terms
    top_terms = ", ".join([term[0].lower() for term in tokens[:5]])
    
    # Construct comprehensive summary
    summary = f"""This text contains {wc} words organized across {sc} sentences, maintaining a {sentiment_desc}. The content demonstrates {complexity}, with primary focus on topics including {top_terms}. The analysis reveals a well-structured composition that effectively communicates its key messages through strategic use of language and thematic elements."""
    
    return summary


# ------------ LOAD PROCESSED DATA ------------- #

def load_processed_text():
    """Loads processed txt or csv from Final_data."""

    folder = "Final_data"
    txt_path = os.path.join(folder, "processed_text.txt")

    # Load processed text
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            return f.read(), "txt"

    # Load processed csv
    csv_path = os.path.join(folder, "processed_csv.csv")
    if os.path.exists(csv_path):
        import pandas as pd # type: ignore
        df = pd.read_csv(csv_path)

        # Combine all string/object columns into a single large text blob
        combined = " ".join(
            df.select_dtypes(include="object")
              .astype(str)
              .fillna("")
              .values.flatten()
        )
        return combined, "csv"

    return None, None


# ------------ TOPIC MODELING ------------- #

def extract_topics(text, n_topics=3):
    """Extract main topics from text using LDA"""
    try:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < n_topics:
            n_topics = max(1, len(sentences) - 1)
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences[:100])  # Limit to 100 sentences
        
        # LDA topic modeling
        if tfidf_matrix.shape[0] < n_topics:
            n_topics = tfidf_matrix.shape[0]
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
        lda.fit(tfidf_matrix)
        
        # Extract top words per topic
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-5:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f"Topic {topic_idx+1}"] = top_words
        
        return topics
    except Exception as e:
        return {"Error": [str(e)]}


def readability_score(text):
    """Calculate simple readability metrics"""
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    
    if len(sentences) == 0:
        return 0
    
    avg_words_per_sentence = len(words) / len(sentences)
    avg_letters_per_word = sum(len(w) for w in words) / max(len(words), 1)
    
    # Flesch-Kincaid Grade Level approximation
    flesch_kincaid = 0.39 * avg_words_per_sentence + 11.8 * (avg_letters_per_word / 5) - 15.59
    return max(0, round(flesch_kincaid, 1))
