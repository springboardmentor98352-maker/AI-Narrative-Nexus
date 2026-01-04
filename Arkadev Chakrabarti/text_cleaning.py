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

# Enhanced text cleaning function
def clean_text(text, remove_stopwords=True, lemmatize=True, min_word_length=2):
    """
    Enhanced text cleaning with configurable options.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Filter by minimum word length
    tokens = [word for word in tokens if len(word) >= min_word_length]
    
    # Lemmatization if requested
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join cleaned tokens
    cleaned_text = " ".join(tokens)
    return cleaned_text