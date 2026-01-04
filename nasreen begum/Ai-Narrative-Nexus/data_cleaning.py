import re
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:

    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove non-ascii (emojis, symbols)
    text = text.encode("ascii", "ignore").decode()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_large_text(text: str, chunk_size: int = 50000) -> str:

    cleaned_chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        cleaned_chunks.append(clean_text(chunk))

    return " ".join(cleaned_chunks)


def split_sentences(text):
    

    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 8]

def summarize_text(text: str, top_n: int = 4) -> str:

    sentences = split_sentences(text)

    if len(sentences) <= top_n:
        return text  

    vectorizer = TfidfVectorizer(max_features=5000) 
    tfidf_matrix = vectorizer.fit_transform(sentences)

    scores = tfidf_matrix.sum(axis=1).A1
    top_indexes = scores.argsort()[-top_n:][::-1]

    summary = " ".join([sentences[i] for i in top_indexes])
    return summary



def get_cosine_similarity(text1: str, text2):

    if isinstance(text2, str) and "\n" in text2:
        comp_list = [line.strip() for line in text2.splitlines() if line.strip()]
    elif isinstance(text2, list):
        comp_list = [c for c in text2 if str(c).strip()]
    else:
        comp_list = [text2]

    texts = [text1] + comp_list

    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(texts)

    result = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    if len(result) == 1:
        return float(result[0])

    return {comp_list[i]: float(result[i]) for i in range(len(comp_list))}
