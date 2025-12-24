import streamlit as st
import pandas as pd
import nltk

from data_cleaning import clean_text, get_cosine_similarity, summarize_text
from sentiment_analysis import analyze_sentiment
from topic_modeling import get_topics


nltk.download("stopwords")

st.title("🧠 Dynamic Text Analysis Platform")




option = st.radio("Choose input method:", ["Upload File", "Paste Text"])
text_data = ""

if option == "Upload File":
    file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)

            possible_cols = ["text", "description", "content", "message", "body", "comment"]
            col = next((c for c in possible_cols if c in df.columns), None)

            if col:
                st.success(f"Detected text column: {col}")
                text_data = " ".join(df[col].astype(str).tolist())
                st.text_area("Extracted Text Preview", text_data[:2000], height=200)
            else:
                st.error("No valid text column found.")
        else:
            text_data = file.read().decode("utf-8", errors="replace")
            st.text_area("Uploaded Text Preview", text_data[:2000], height=200)

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

        st.success("✔ Text cleaned & summarized!")


if "cleaned" in st.session_state:
    st.subheader("📄 Cleaned Text")
    st.text_area("Cleaned Output", st.session_state["cleaned"], height=300)

    st.subheader("📘 Summary")
    st.text_area("Summary Output", st.session_state["summary"], height=200)

    cleaned_text = st.session_state["cleaned"]

    st.metric("Word Count", len(cleaned_text.split()))
    st.metric("Character Count", len(cleaned_text))


st.subheader("📌 Cosine Similarity")

compare_text = st.text_area("Enter text to compare with cleaned text:", height=150)

if st.button("Calculate Cosine Similarity"):
    if "cleaned" not in st.session_state:
        st.error("⚠ Please analyze text before calculating similarity.")
    elif not compare_text.strip():
        st.warning("⚠ Enter some comparison text.")
    else:
        score = get_cosine_similarity(st.session_state["cleaned"], compare_text)
        st.success(f"Cosine Similarity Score: **{score:.4f}**")

        st.subheader("💬 Sentiment Analysis")

if "cleaned" in st.session_state:
    sentiment, score = analyze_sentiment(st.session_state["cleaned"])
    st.success(f"Sentiment: **{sentiment}**")
    st.write(f"Polarity Score: {score:.2f}")

    st.subheader("📂 Topic Modeling")

model_choice = st.selectbox("Choose Topic Model", ["LDA", "NMF"])

if st.button("Extract Topics"):
    if "cleaned" not in st.session_state:
        st.warning("Please analyze text first")
    else:
        topics = get_topics(
            st.session_state["cleaned"],
            model_type=model_choice
        )

        for i, topic in enumerate(topics, 1):
            st.write(f"**Topic {i}:** {topic}")



