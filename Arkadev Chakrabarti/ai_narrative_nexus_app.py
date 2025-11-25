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

# PDF Support
try:
    import PyPDF2
except ImportError:
    st.stop()

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    for r in ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']:
        nltk.download(r, quiet=True)

download_nltk_data()

# ========================
# File Readers
# ========================

def read_txt(file): 
    return file.read().decode("utf-8")

def read_csv(file): 
    return pd.read_csv(file)

def read_docx(file):
    return "\n".join([p.text for p in Document(file).paragraphs])

def read_pdf(file):
    text = ""
    file.seek(0)
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    except: pass

    return text.strip() if text.strip() else "[No text extracted from PDF]"

# ========================
# Text Cleaning
# ========================

def clean_text(text, remove_stopwords=True, lemmatize=True, min_word_length=2):
    if not isinstance(text, str) or not text.strip():
        return "", []

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]

    tokens = [t for t in tokens if len(t) >= min_word_length]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens), tokens

# ========================
# Sentiment
# ========================

def analyze_sentiment(text):
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    if pol > 0.1: cat = "Positive"
    elif pol < -0.1: cat = "Negative"
    else: cat = "Neutral"
    return {"polarity": pol, "subjectivity": blob.sentiment.subjectivity, "category": cat}

# ========================
# Streamlit App
# ========================

st.set_page_config(page_title="Narrative Nexus", layout="wide")
st.title("Narrative Nexus")
st.write("Dynamic Text Analysis Platform")

# Sidebar
with st.sidebar:
    st.header("Settings & Configuration")
    st.subheader("Text Cleaning Options")
    remove_stopwords = st.checkbox("Remove stopwords", True)
    apply_lemmatization = st.checkbox("Apply lemmatization", True)
    min_word_len = st.slider("Min word length", 1, 10, 2)
    top_n_words = st.slider("Top N frequent words to show", 5, 50, 20)

# Session State
for k in ['text_data', 'cleaned_text', 'tokens']:
    if k not in st.session_state:
        st.session_state[k] = "" if k != 'tokens' else []

tab1, tab2 = st.tabs(["Upload File", "Enter Text Manually"])

# Upload
with tab1:
    file = st.file_uploader("Upload file", type=["txt","csv","docx","pdf"])
    if file:
        ext = file.name.split(".")[-1].lower()
        st.success(f"Uploaded: {file.name}")

        if ext == "txt": st.session_state.text_data = read_txt(file)
        elif ext == "csv":
            df = read_csv(file)
            st.dataframe(df)
            cols = df.select_dtypes(include='object').columns
            if len(cols)>0:
                col = st.selectbox("Text column", cols)
                st.session_state.text_data = " ".join(df[col].dropna().astype(str))
        elif ext == "docx": st.session_state.text_data = read_docx(file)
        elif ext == "pdf":
            with st.spinner("Extracting PDF..."):
                st.session_state.text_data = read_pdf(file)

        if st.session_state.text_data:
            st.text_area("Extracted Text", st.session_state.text_data, height=300)

# Manual
with tab2:
    txt = st.text_area("Enter text here manually:", height=300)
    if txt.strip():
        st.session_state.text_data = txt
        st.success("Text loaded!")

# ========================
# ANALYSIS
# ========================

if st.session_state.text_data and st.session_state.text_data.strip():
    st.markdown("---")
    st.header("Text Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Clean & Analyze Text", type="primary", use_container_width=True):
            with st.spinner("Processing text..."):
                cleaned_str, token_list = clean_text(
                    st.session_state.text_data,
                    remove_stopwords,
                    apply_lemmatization,
                    min_word_len
                )
                st.session_state.cleaned_text = cleaned_str
                st.session_state.tokens = token_list

                st.success("Analysis Complete!")

                # Token Stats
                total = len(token_list)
                unique = len(set(token_list))

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Tokens", total)
                c2.metric("Unique Tokens", unique)
                c3.metric("Lexical Diversity", f"{unique/total:.3f}" if total else "0")
                # Show token preview
                with st.expander("View Tokens (first 100)"):
                    st.write(token_list[:100])
                # Sentence Stats
                sentences = sent_tokenize(st.session_state.text_data)
                words_per_sent = [len(word_tokenize(s)) for s in sentences]
                avg_words = sum(words_per_sent) / len(words_per_sent) if words_per_sent else 0

                s1, s2, s3 = st.columns(3)
                s1.metric("Total Sentences", len(sentences))
                s2.metric("Avg Words/Sentence", f"{avg_words:.1f}")
                s3.metric("Longest Sentence", max(words_per_sent) if words_per_sent else 0)

                # MOST FREQUENT WORDS
                st.subheader(f"Top {top_n_words} Most Frequent Words")
                word_counts = Counter(token_list)
                common = word_counts.most_common(top_n_words)

                if common:
                    df_freq = pd.DataFrame(common, columns=["Word", "Frequency"])

                    # Interactive Plotly Chart
                    fig = px.bar(
                        df_freq, x="Frequency", y="Word", orientation='h',
                        text="Frequency", color="Frequency",
                        color_continuous_scale="Viridis",
                        title=f"Top {top_n_words} Words"
                    )
                    fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    # Downloadable Table
                    with st.expander("View & Download Frequency Table"):
                        st.dataframe(df_freq, use_container_width=True)
                        csv = df_freq.to_csv(index=False).encode()
                        st.download_button(
                            "Download CSV",
                            csv,
                            "word_frequency.csv",
                            "text/csv"
                        )
                else:
                    st.info("No tokens after cleaning.")

    with col2:
        st.subheader("Additional Tools")
        if st.button("Sentiment Analysis", type="secondary", use_container_width=True):
            sent = analyze_sentiment(st.session_state.text_data)
            if sent['category'] == "Positive":
                st.success(f"Positive")
            elif sent['category'] == "Negative":
                st.error(f"Negative")
            else:
                st.info(f"Neutral")

            st.metric("Polarity", f"{sent['polarity']:.3f}")
            st.metric("Subjectivity", f"{sent['subjectivity']:.3f}")

        st.markdown("---")
        if st.session_state.cleaned_text:
            st.download_button(
                "Download Cleaned Text",
                st.session_state.cleaned_text,
                "cleaned_text.txt",
                "text/plain"
            )

st.markdown("---")
st.markdown("**Narrative Nexus** — Dynamic Text Analysis Tool | Developed by Arkadev" )