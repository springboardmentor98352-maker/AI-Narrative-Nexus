import streamlit as st
import pandas as pd
from docx import Document
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from heapq import nlargest
import numpy as np
import base64
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS

# Transformers for Abstractive Summarization
try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # Will handle in the function

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    for r in ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']:
        nltk.download(r, quiet=True)

download_nltk_data()

# ========================
# Summarization Functions
# ========================

def normalize_topic_words(words):
    """
    Ensures topic words are always a list of strings.
    Handles strings, lists, tuples, numpy arrays, etc.
    """
    if not words:
        return []

    if isinstance(words, str):
        return [words]

    if hasattr(words, "__iter__"):
        # Handle (word, score) tuples or mixed structures
        return [
            w if isinstance(w, str) else str(w[0])
            for w in words
        ]

    return [str(words)]

def build_themes_text(topics_data):
    """
    Formats topic data into readable theme text.
    """
    themes = []

    for i, (words, *_ ) in enumerate(topics_data):
        normalized_words = normalize_topic_words(words)
        theme_line = f"Topic {i + 1}: {', '.join(normalized_words)}"
        themes.append(theme_line)

    return "\n".join(themes)

def prepare_summary_input(themes_text, full_text, max_context_len=1500):
    """
    Combines themes and context into a single summarization input.
    """
    context = full_text[:max_context_len]
    return f"Themes:\n{themes_text}\n\nContext:\n{context}"

def extractive_summarize(text, num_sentences=3):
    if not text.strip():
        return "No text to summarize."
    
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    stop_words = set(stopwords.words("english"))
    word_freq = Counter(word.lower() for word in word_tokenize(text) if word.lower() not in stop_words and word.isalpha())
    
    sentence_scores = {}
    for sent in sentences:
        words = word_tokenize(sent.lower())
        score = sum(word_freq.get(w, 0) for w in words if w not in stop_words and w.isalpha())
        sentence_scores[sent] = score / (len(words) + 1)
    
    top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_sentences = sorted(top_sentences, key=lambda x: text.index(x))
    
    return ' '.join(summary_sentences)

@st.cache_resource
def get_summarizer():
    if pipeline is None:
        raise ImportError("Transformers library not installed.")
    return pipeline("summarization", model="facebook/bart-large-cnn")

def abstractive_summarize(text, max_length=130, min_length=30):
    if not text.strip():
        return "No text to summarize."
    
    try:
        summarizer = get_summarizer()
        if len(text) > 4000:
            text = text[:4000] + "..."
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    except ImportError:
        return "Install 'transformers' and 'torch' for abstractive summarization."
    except Exception as e:
        return f"Error: {str(e)}"
