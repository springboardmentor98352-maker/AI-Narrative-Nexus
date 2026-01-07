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

# ===============================================
# Topic Modeling with Document-Topic Distribution
# ===============================================

def perform_topic_modeling(documents, num_topics=5, algorithm='LDA', top_words=10):
    if not documents or len(documents) == 0:
        st.warning("No documents available for topic modeling.")
        return [], None, None

    n_docs = len(documents)

    # Dynamically adjust min_df and max_df
    if n_docs < 10:
        min_df = 1          # Allow words appearing in just 1 document
        max_df = 1.0        # No upper limit
    elif n_docs < 50:
        min_df = 1
        max_df = 0.8
    else:
        min_df = 2
        max_df = 0.95

    try:
        if algorithm == 'LDA':
            vectorizer = CountVectorizer(
                min_df=min_df,
                max_df=max_df,
                stop_words='english'
            )
            dtm = vectorizer.fit_transform(documents)
            model = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=20
            )
        elif algorithm == 'NMF':
            vectorizer = TfidfVectorizer(
                min_df=min_df,
                max_df=max_df,
                stop_words='english'
            )
            dtm = vectorizer.fit_transform(documents)
            model = NMF(
                n_components=num_topics,
                random_state=42,
                init='nndsvd',
                max_iter=300
            )
        else:
            raise ValueError("Unsupported algorithm.")

        # If after filtering, we have no features → abort gracefully
        if dtm.shape[1] == 0:
            st.warning("Not enough unique words after filtering to perform topic modeling. Try with longer text.")
            return [], None, None

        model.fit(dtm)
        doc_topic_dist = model.transform(dtm)
        topic_prevalence = np.mean(doc_topic_dist, axis=0)

        feature_names = vectorizer.get_feature_names_out()

        topics = []
        for topic_idx, topic in enumerate(model.components_):
            sorted_indices = topic.argsort()[::-1]
            top_features = [feature_names[i] for i in sorted_indices[:top_words]]
            top_scores = [topic[i] for i in sorted_indices[:top_words]]
            max_score = top_scores[0] if top_scores else 1
            normalized_scores = [score / max_score if max_score > 0 else 0 for score in top_scores]
            topics.append((topic_idx, top_features, top_scores, normalized_scores))

        return topics, topic_prevalence, vectorizer

    except ValueError as e:
        if "max_df corresponds to < documents than min_df" in str(e):
            st.error("Text is too short or has too few repeated words for topic modeling with current settings.")
            st.info("Tip: Try uploading a longer document (at least 10–15 sentences) for better topic results.")
        else:
            st.error(f"Error during topic modeling: {str(e)}")
        return [], None, None
