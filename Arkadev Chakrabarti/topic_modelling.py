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

# =====================================
# Topic Modeling with Importance Scores
# =====================================

def perform_topic_modeling(documents, num_topics=5, algorithm='LDA', top_words=10):
    if not documents:
        return []

    if algorithm == 'LDA':
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(documents)
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    elif algorithm == 'NMF':
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(documents)
        model = NMF(n_components=num_topics, random_state=42, init='nndsvd')  # Better initialization for stability
    else:
        raise ValueError("Unsupported algorithm. Choose 'LDA' or 'NMF'.")

    model.fit(dtm)
    
    feature_names = vectorizer.get_feature_names_out()
    
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        # Get indices sorted by importance descending
        sorted_indices = topic.argsort()[::-1]
        top_features = [feature_names[i] for i in sorted_indices[:top_words]]
        top_scores = [topic[i] for i in sorted_indices[:top_words]]
        
        # Normalize scores for better visualization (max = 1)
        max_score = top_scores[0] if top_scores else 1
        normalized_scores = [score / max_score for score in top_scores]
        
        topics.append((topic_idx, top_features, top_scores, normalized_scores))
    
    return topics, vectorizer, model
