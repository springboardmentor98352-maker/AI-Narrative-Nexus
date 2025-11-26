import streamlit as st
import pandas as pd
import nltk
from data_cleaning import clean_text

nltk.download("stopwords")

st.title("🧠 Dynamic Text Analysis Platform")

option = st.radio("Choose input method:", ["Upload File", "Paste Text"])

text_data = ""
cleaned_text = ""


if option == "Upload File":
    uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

            possible_cols = ["description", "text", "content", "title"]
            text_col = next((col for col in possible_cols if col in df.columns), None)

            if text_col:
                st.success(f"Detected text column: {text_col}")
                text_data = " ".join(df[text_col].astype(str).tolist())
                st.text_area("Extracted Text:", text_data[:1500], height=200)
            else:
                st.error("❌ No readable text column found in the CSV.")

        else:  
            text_data = uploaded_file.read().decode("utf-8")
            st.text_area("Raw Text:", text_data[:3000], height=200)


else:
    text_data = st.text_area("Paste your text here:", "", height=200)


if st.button("🔍 Analyze Text"):

    if not text_data.strip():
        st.warning("Please upload or paste text before analyzing.")
    else:

        cleaned_text = clean_text(text_data)

        st.session_state["cleaned_full"] = cleaned_text

        st.success("✔ Text cleaned successfully! Click below to see full output.")

if "cleaned_full" in st.session_state:
    if st.button("✨ Show Full Cleaned Data"):

        st.subheader("📄 Full Cleaned Output")
        st.text_area(
            "Cleaned Text:",
            st.session_state["cleaned_full"],
            height=300
        )

        word_count = len(st.session_state["cleaned_full"].split())
        char_count = len(st.session_state["cleaned_full"])

        st.write(f"### 🧮 Word Count: {word_count}")
        st.write(f"### 🔤 Character Count: {char_count}")