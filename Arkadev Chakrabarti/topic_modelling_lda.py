import streamlit as st
import pandas as pd
from docx import Document
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    resources = ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass
download_nltk_data()

def perform_lda(texts, n_topics=5, n_words=10):
    """
    Perform Latent Dirichlet Allocation for topic modeling.
    """
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Train LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='online'
    )
    lda_output = lda_model.fit_transform(doc_term_matrix)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract topics
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_num': topic_idx + 1,
            'words': top_words,
            'weights': [topic[i] for i in top_words_idx]
        })
    
    return lda_model, lda_output, topics, vectorizer

def perform_nmf(texts, n_topics=5, n_words=10):
    """
    Perform Non-negative Matrix Factorization for topic modeling.
    """
    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # Train NMF model
    nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=200)
    nmf_output = nmf_model.fit_transform(tfidf_matrix)
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Extract topics
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_num': topic_idx + 1,
            'words': top_words,
            'weights': [topic[i] for i in top_words_idx]
        })
    
    return nmf_model, nmf_output, topics, tfidf_vectorizer