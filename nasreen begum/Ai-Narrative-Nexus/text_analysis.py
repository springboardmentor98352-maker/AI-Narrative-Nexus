import streamlit as st
import pandas as pd
import nltk

from data_cleaning import clean_text, get_cosine_similarity

nltk.download("stopwords")

st.title("🧠 Dynamic Text Analysis Platform")

option = st.radio("Choose input method:", ["Upload File", "Paste Text"])
text_data = ""

if option == "Upload File":
    file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
            possible_cols = ["text","description","content","message","body","comment"]
            col = next((c for c in possible_cols if c in df.columns), None)
            if col:
                st.success(f"Detected text column: {col}")
                text_data = " ".join(df[col].astype(str).tolist())
                st.text_area("Extracted Text (Preview)", text_data[:2000], height=200)
            else:
                st.error("No valid text column found.")
        else:
            text_data = file.read().decode("utf-8", errors="replace")
            st.text_area("Uploaded Text Preview", text_data[:2000], height=200)

else:
    text_data = st.text_area("Paste your text", height=200)

if st.button("🔍 Analyze Text"):
    if not text_data.strip():
        st.warning("Enter or upload text before analyzing.")
    else:
        cleaned = clean_text(text_data)
        st.session_state["cleaned"] = cleaned
        st.success("✔ Cleaning completed!")

if "cleaned" in st.session_state:
    st.subheader("📄 Cleaned Text")
    st.text_area("Cleaned Output", st.session_state["cleaned"], height=300)
    st.metric("Word Count", len(st.session_state["cleaned"].split()))
    st.metric("Character Count", len(st.session_state["cleaned"]))

    st.subheader("📌 Cosine Similarity")
    if st.button("Calculate Cosine Similarity"):
        score = get_cosine_similarity(st.session_state["cleaned"], st.session_state["cleaned"])
        st.success(f"Cosine Similarity: {score:.4f}")
