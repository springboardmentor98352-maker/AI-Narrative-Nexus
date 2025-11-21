import streamlit as st
from textblob import TextBlob
import numpy as np

def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob.
    Returns polarity (-1 to 1) and subjectivity (0 to 1).
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Categorize sentiment
    if polarity > 0.1:
        category = "Positive"
    elif polarity < -0.1:
        category = "Negative"
    else:
        category = "Neutral"
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'category': category
    }
