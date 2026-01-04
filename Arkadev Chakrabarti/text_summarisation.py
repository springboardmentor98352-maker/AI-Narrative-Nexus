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
        words = word_tokenize(sent)
        score = sum(word_freq.get(word.lower(), 0) for word in words if word.lower() not in stop_words and word.isalpha())
        sentence_scores[sent] = score / len(words) if words else 0
    
    top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # Sort top sentences by their order in the original text
    summary_sentences = sorted(top_sentences, key=lambda x: text.index(x))
    
    return ' '.join(summary_sentences)

@st.cache_resource
def get_summarizer():
    if pipeline is None:
        raise ImportError("Transformers library not installed.")
    return pipeline("summarization", model="facebook/bart-large-cnn")  # More reliable default model

def abstractive_summarize(text, max_length=130, min_length=30):
    if not text.strip():
        return "No text to summarize."
    
    try:
        summarizer = get_summarizer()
        if len(text) > 4000:
            text = text[:4000] + "... (truncated for summarization)"
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    except ImportError:
        return "Please install the 'transformers' library to use abstractive summarization. Run 'pip install transformers torch'."
    except Exception as e:
        return f"Error in abstractive summarization: {str(e)}"
