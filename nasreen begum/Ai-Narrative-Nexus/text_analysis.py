import streamlit as st 
import pandas as pd
import nltk
from textblob import TextBlob
import plotly.graph_objects as go
from PyPDF2 import PdfReader
import xml.etree.ElementTree as ET

from data_cleaning import clean_text, get_cosine_similarity, summarize_text
from topic_modeling import get_topics

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")
nltk.download("stopwords")

st.title("🧠 Dynamic Text Analysis Platform")


if "file_texts" not in st.session_state:
    st.session_state["file_texts"] = []

if "combined_text" not in st.session_state:
    st.session_state["combined_text"] = ""

if "show_sentiment" not in st.session_state:
    st.session_state["show_sentiment"] = False

if "show_cosine" not in st.session_state:
    st.session_state["show_cosine"] = False

if "show_filewise" not in st.session_state:
    st.session_state["show_filewise"] = False


option = st.radio("Choose input method:", ["Upload File", "Paste Text"])

if option == "Upload File":
    files = st.file_uploader(
        "Upload files",
        type=["csv", "txt", "pdf", "xml"],
        accept_multiple_files=True
    )

    if files:
        all_texts = []

        for file in files:
            file_text = ""

            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                possible_cols = ["text", "description", "content", "message", "body", "comment"]
                col = next((c for c in possible_cols if c in df.columns), None)
                if col:
                    file_text = " ".join(df[col].astype(str))

            elif file.name.endswith(".txt"):
                file_text = file.read().decode("utf-8", errors="ignore")

            elif file.name.endswith(".pdf"):
                reader = PdfReader(file)
                file_text = " ".join(page.extract_text() or "" for page in reader.pages)

            elif file.name.endswith(".xml"):
                tree = ET.parse(file)
                root = tree.getroot()
                file_text = " ".join(elem.text for elem in root.iter() if elem.text)

            if file_text.strip():
                all_texts.append(file_text)

        st.session_state["file_texts"] = all_texts
        st.session_state["combined_text"] = "\n\n".join(all_texts)

        st.text_area(
            "📌 Combined Text Preview (All Uploaded Files)",
            st.session_state["combined_text"][:2000],
            height=200
        )

else:
    st.text_area("Paste your text", height=200)


if st.button("🔍 Analyze Text"):
    if not st.session_state["file_texts"]:
        st.warning("Please upload files.")
    else:
        st.session_state["show_filewise"] = True
        st.success("✔ Files analyzed successfully!")


if st.session_state.get("show_filewise") and st.session_state["file_texts"]:
    st.subheader("📂 File-wise Cleaned Text & Summary")

    for idx, raw_text in enumerate(st.session_state["file_texts"], start=1):
        cleaned_file = clean_text(raw_text)
        summary_file = summarize_text(raw_text)

        st.markdown(f"### 📄 File {idx}")
        st.text_area(f"🧹 Cleaned Text - File {idx}", cleaned_file, height=200)
        st.text_area(f"📝 Summary - File {idx}", summary_file, height=150)

        col1, col2 = st.columns(2)
        col1.metric("Word Count", len(cleaned_file.split()))
        col2.metric("Character Count", len(cleaned_file))


st.subheader("📌 Cosine Similarity")
compare_text = st.text_area("Enter text to compare with cleaned text:", height=150)

if st.button("Calculate Cosine Similarity"):
    if not st.session_state["file_texts"]:
        st.error("⚠ Upload files first.")
    elif not compare_text.strip():
        st.warning("⚠ Enter comparison text.")
    else:
        st.session_state["show_cosine"] = True

if st.session_state.get("show_cosine") and st.session_state["file_texts"]:
    st.subheader("📌 Cosine Similarity Per File")
    for idx, raw_text in enumerate(st.session_state["file_texts"], start=1):
        score = get_cosine_similarity(clean_text(raw_text), compare_text)
        st.markdown(f"**File {idx} Similarity:** {score:.4f}")

if st.button("💬 Show Sentiment Analysis"):
    st.session_state["show_sentiment"] = True

if st.session_state.get("show_sentiment") and st.session_state["file_texts"]:
    st.subheader("💬 Sentiment Analysis")

    cols = st.columns(len(st.session_state["file_texts"]))

    for idx, (col, raw_text) in enumerate(
        zip(cols, st.session_state["file_texts"]), start=1
    ):
        with col:
            cleaned_file = clean_text(raw_text)

            polarity_score = TextBlob(cleaned_file).sentiment.polarity

            
            if -0.05 <= polarity_score <= 0.05:
                polarity_score = 0

            sentences = cleaned_file.split(".")
            pos_strength = neg_strength = neu_count = 0
            pos_count = neg_count = 0

            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                p = TextBlob(s).sentiment.polarity

                if -0.05 <= p <= 0.05:
                    neu_count += 1
                elif p > 0:
                    pos_strength += p
                    pos_count += 1
                else:
                    neg_strength += abs(p)
                    neg_count += 1

            pos_value = pos_strength / pos_count if pos_count else 0
            neg_value = neg_strength / neg_count if neg_count else 0
            neu_value = neu_count / max(len(sentences), 1)

            sentiments = ["Positive 😊", "Neutral 😐", "Negative 😞"]
            values = [pos_value, neu_value, neg_value]
            colors = ["#1f77b4", "#ff9800", "#e53935"]

            fig = go.Figure(
                go.Bar(
                    x=sentiments,
                    y=values,
                    marker_color=colors
                )
            )

            fig.add_annotation(
                text=f"Polarity Score: {polarity_score:.3f}",
                x=0.5,
                y=1.15,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="black")
            )

            fig.update_layout(
                title=f"File {idx}",
                height=320,
                yaxis_range=[0, 1],
                margin=dict(l=10, r=10, t=60, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)

st.subheader("📂 Topic Modeling Per File")

if st.session_state["file_texts"] and st.button("Extract Topics Per File"):
    for idx, raw_text in enumerate(st.session_state["file_texts"], start=1):
        topic = get_topics(clean_text(raw_text), model_type="NMF")[0]
        st.markdown(f"### 📄 File {idx}")
        st.write(f"**Topic {idx} (NMF):** {topic}")
