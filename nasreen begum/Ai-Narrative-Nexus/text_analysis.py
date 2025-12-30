import streamlit as st
import pandas as pd
import nltk
from textblob import TextBlob
import plotly.graph_objects as go
from PyPDF2 import PdfReader
import xml.etree.ElementTree as ET

from data_cleaning import clean_text, get_cosine_similarity, summarize_text
from sentiment_analysis import analyze_sentiment
from topic_modeling import get_topics

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

nltk.download("stopwords")

st.title("🧠 Dynamic Text Analysis Platform")

if "file_texts" not in st.session_state:
    st.session_state["file_texts"] = []
if "cleaned" not in st.session_state:
    st.session_state["cleaned"] = ""
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "show_sentiment" not in st.session_state:
    st.session_state["show_sentiment"] = False
if "show_cosine" not in st.session_state:
    st.session_state["show_cosine"] = False
if "show_filewise" not in st.session_state:
    st.session_state["show_filewise"] = False  # NEW: control when to show file-wise

option = st.radio("Choose input method:", ["Upload File", "Paste Text"])
text_data = ""
all_texts = []

if option == "Upload File":
    files = st.file_uploader(
        "Upload files",
        type=["csv", "txt", "pdf", "xml"],
        accept_multiple_files=True
    )

    if files:
        for file in files:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                possible_cols = ["text", "description", "content", "message", "body", "comment"]
                col = next((c for c in possible_cols if c in df.columns), None)
                if col:
                    all_texts.append(" ".join(df[col].astype(str)))
            elif file.name.endswith(".txt"):
                all_texts.append(file.read().decode("utf-8", errors="ignore"))
            elif file.name.endswith(".pdf"):
                reader = PdfReader(file)
                pdf_text = " ".join(page.extract_text() or "" for page in reader.pages)
                all_texts.append(pdf_text)
            elif file.name.endswith(".xml"):
                tree = ET.parse(file)
                root = tree.getroot()
                xml_text = " ".join(elem.text for elem in root.iter() if elem.text)
                all_texts.append(xml_text)

        st.session_state["file_texts"] = all_texts
        text_data = " ".join(all_texts)
        st.text_area("Combined Text Preview", text_data[:2000], height=200)
else:
    text_data = st.text_area("Paste your text", height=200)

if st.button("🔍 Analyze Text"):
    if not text_data.strip():
        st.warning("Please enter or upload text.")
    else:
        cleaned = clean_text(text_data)
        summary = summarize_text(text_data)
        st.session_state["cleaned"] = cleaned
        st.session_state["summary"] = summary
        st.session_state["show_filewise"] = True  # NEW: show file-wise after analyze
        st.success("✔ Text cleaned & summarized!")

if st.session_state["cleaned"]:
    st.subheader("📄 Cleaned Text")
    st.text_area("Cleaned Output", st.session_state["cleaned"], height=200)
    st.subheader("📘 Summary")
    st.text_area("Summary Output", st.session_state["summary"], height=150)

if st.session_state["cleaned"]:
    st.markdown("### 📊 Total Metrics")
    col1, col2 = st.columns(2)
    col1.metric("**Total Word Count**", len(st.session_state["cleaned"].split()))
    col2.metric("**Total Character Count**", len(st.session_state["cleaned"]))

if st.session_state.get("show_filewise") and st.session_state["file_texts"]:
    st.subheader("📂 File-wise Cleaned Text & Summary")
    for idx, raw_text in enumerate(st.session_state["file_texts"], start=1):
        cleaned_file = clean_text(raw_text)
        summary_file = summarize_text(raw_text)
        st.markdown(f"### 📄 File {idx}")
        st.text_area(f"Cleaned Text - File {idx}", cleaned_file, height=200)
        st.text_area(f"Summary - File {idx}", summary_file, height=150)
        col1, col2 = st.columns(2)
        col1.metric(f"**Word Count - File {idx}**", len(cleaned_file.split()))
        col2.metric(f"**Character Count - File {idx}**", len(cleaned_file))

st.subheader("📌 Cosine Similarity")
compare_text = st.text_area("Enter text to compare with cleaned text:", height=150)

if st.button("Calculate Cosine Similarity"):
    if not st.session_state["file_texts"]:
        st.error("⚠ Please upload and analyze files first.")
    elif not compare_text.strip():
        st.warning("⚠ Enter some comparison text.")
    else:
        st.session_state["show_cosine"] = True

if st.session_state.get("show_cosine") and st.session_state["file_texts"]:
    st.subheader("📌 Cosine Similarity Per File")
    for idx, raw_text in enumerate(st.session_state["file_texts"], start=1):
        cleaned_file = clean_text(raw_text)
        score = get_cosine_similarity(cleaned_file, compare_text)
        st.markdown(f"**File {idx} Cosine Similarity:** {score:.4f}")

if st.button("💬 Show Sentiment Analysis"):
    st.session_state["show_sentiment"] = True

if st.session_state.get("show_sentiment") and st.session_state["file_texts"]:
    st.subheader("💬 Sentiment Analysis Per File")
    for idx, raw_text in enumerate(st.session_state["file_texts"], start=1):
        cleaned_file = clean_text(raw_text)
        sentences = cleaned_file.split(".")
        pos_strength = neg_strength = neu_count = 0
        pos_count = neg_count = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            polarity = TextBlob(s).sentiment.polarity
            if polarity > 0:
                pos_strength += polarity
                pos_count += 1
            elif polarity < 0:
                neg_strength += abs(polarity)
                neg_count += 1
            else:
                neu_count += 1
        pos_value = pos_strength / pos_count if pos_count else 0
        neg_value = neg_strength / neg_count if neg_count else 0
        neu_value = neu_count / max(len(sentences), 1)
        st.markdown(f"### 📄 File {idx} Sentiment")
        st.success(
            f"Avg Positive: {pos_value:.2f}, "
            f"Avg Negative: {neg_value:.2f}, "
            f"Neutral: {neu_value:.2f}"
        )
        sentiments = ["Positive 😊", "Neutral 😐", "Negative 😞"]
        values = [pos_value, neu_value, neg_value]
        fig = go.Figure([go.Bar(x=sentiments, y=values)])
        fig.update_layout(
            title=f"Sentiment Distribution - File {idx}",
            yaxis_title="Polarity Strength (0–1)",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig)

st.subheader("📂 Topic Modeling Per File")
if st.session_state["file_texts"] and st.button("Extract Topics Per File"):
    for idx, raw_text in enumerate(st.session_state["file_texts"], start=1):
        cleaned_file = clean_text(raw_text)
        nmf_topic = get_topics(cleaned_file, model_type="NMF")[0]
        st.markdown(f"### 📄 File {idx}")
        st.write(f"**Topic {idx} (NMF):** {nmf_topic}")
