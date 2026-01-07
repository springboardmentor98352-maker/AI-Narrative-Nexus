import streamlit as st
import pandas as pd
from docx import Document
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from heapq import nlargest
import numpy as np
import base64
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS

# WordCloud Support (fallback to matplotlib if wordcloud not installed)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ========================
# Word Cloud Utilities
# ========================

STOP_WORDS = frozenset(STOPWORDS)


def build_word_frequencies(tokens, min_freq=2):
    """
    Efficiently build word frequencies from large token lists.
    Filters stopwords, short words, non-alpha tokens.
    """

    counter = Counter()

    for token in tokens:
        token = token.lower()

        if (
            token.isalpha()
            and len(token) >= 3
            and token not in STOP_WORDS
        ):
            counter[token] += 1

    # Remove rare/noisy words
    return {
        word: count
        for word, count in counter.items()
        if count >= min_freq
    }


def generate_wordcloud(
    tokens,
    width=900,
    height=450,
    max_words=200,
    background_color="white"
):
    """
    Generates a WordCloud image buffer (PNG) optimized for Streamlit.
    Returns BytesIO or None.
    """

    word_freq = build_word_frequencies(tokens)

    if not word_freq:
        return None

    wc = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color=background_color,
        stopwords=STOP_WORDS,
        collocations=False,
        prefer_horizontal=0.9,
        min_word_length=3,
        normalize_plurals=True
    ).generate_from_frequencies(word_freq)

    # Convert to PIL Image
    pil_img = wc.to_image()

    # Write image to memory buffer
    img_buffer = BytesIO()
    pil_img.save(img_buffer, format="PNG", optimize=True)
    img_buffer.seek(0)

    return img_buffer
