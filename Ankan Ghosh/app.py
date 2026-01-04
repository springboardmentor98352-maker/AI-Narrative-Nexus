import streamlit as st
from preprocessing import preprocess_for_summary
from preprocessing import preprocess_for_summary_lemmatized

import pandas as pd
# from backend import read_pdf, read_word, read_excel, read_csv


from collections import Counter
import matplotlib.pyplot as plt


def get_top_keywords(text, top_n=10):
    words = text.lower().split()
    counter = Counter(words)
    return counter.most_common(top_n)



from backend import (
    read_pdf,
    read_excel,
    read_word,
    read_csv,
    analyze_text_stats,
    topic_modeling,
    analyze_sentiment,
    summarize_text,
    generate_wordcloud,
)
from preprocessing import preprocess_text


st.set_page_config(
    page_title="Advanced AI Text Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>

/* FULL PAGE GRADIENT BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* MAIN CONTENT CARD */
.main-card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
    margin-bottom: 25px;
}

/* SECTION TITLES */
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #4f46e5;
    margin-bottom: 10px;
}

/* METRIC CARDS */
.metric-card {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    padding: 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
}

/* BUTTON STYLE */
.stButton > button {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: black;
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
    border: none;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    color: white;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* TEXT AREA */
textarea {
    border-radius: 12px !important;
}

.main-card:empty {
    display: none;
}



</style>
""", unsafe_allow_html=True)


st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.title("🧠 NarrativeNexus: The Dynamic Text Analysis Platform")
st.write("Analyze documents and text using **AI-powered NLP techniques** 🚀")
st.markdown("</div>", unsafe_allow_html=True)


st.sidebar.header("⚙️ Preprocessing Controls")

remove_spaces = st.sidebar.checkbox("Remove extra spaces", value=True)
remove_special = st.sidebar.checkbox("Remove special characters", value=True)
apply_stopwords = st.sidebar.checkbox("Remove stopwords", value=True)
apply_stem = st.sidebar.checkbox("Apply Stemming")
apply_lemma = st.sidebar.checkbox("Apply Lemmatization")

st.sidebar.info("💡 Tip: Prefer lemmatization for better summaries.")


st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("""
<style>
.section-title {
    color: white;
    font-size: 26px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='section-title'>📂 Upload File</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Supported formats: PDF, Word, Excel, TXT, CSV",
    type=["pdf", "docx", "xlsx", "txt","csv"]
)

raw_text = ""

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    with st.spinner("Extracting text..."):
        if ext == "pdf":
            raw_text = read_pdf(uploaded_file)
        elif ext == "docx":
            raw_text = read_word(uploaded_file)
        elif ext == "xlsx":
            raw_text = read_excel(uploaded_file)
        elif ext == "csv":
            raw_text = read_csv(uploaded_file)
        elif ext == "txt":
            raw_text = uploaded_file.read().decode("utf-8")
        

    st.success("✅ File processed successfully!")
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📝 Text Input</div>", unsafe_allow_html=True)

user_input = st.text_area(
    "Paste or edit your text here",
    value=raw_text,
    height=260
)

analyze_btn = st.button("🚀 Analyze Text", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


if analyze_btn:

    if not user_input.strip():
        st.warning("⚠️ Please enter text first.")
        st.stop()

    processed_text = preprocess_text(
        user_input,
        remove_space=remove_spaces,
        remove_special=remove_special,
        stopword_flag=apply_stopwords,
        stem_flag=apply_stem,
        lemma_flag=apply_lemma,
    )

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔍 Processed Text</div>", unsafe_allow_html=True)
    st.text_area("Processed Output", processed_text, height=160)
    st.markdown("</div>", unsafe_allow_html=True)


    char_c, word_c, line_c, sent_c = analyze_text_stats(user_input)

    st.markdown("<div class='section-title'>📊 Text Statistics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"<div class='metric-card'><h3>{char_c}</h3><p>Characters</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><h3>{word_c}</h3><p>Words</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><h3>{line_c}</h3><p>Lines</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><h3>{sent_c}</h3><p>Sentences</p></div>", unsafe_allow_html=True)


    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>😊 Sentiment Analysis</div>", unsafe_allow_html=True)
    polarity, sentiment = analyze_sentiment(processed_text)
    st.success(f"Sentiment: **{sentiment}** | Polarity: **{round(polarity, 3)}**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧩 Topic Modeling</div>", unsafe_allow_html=True)
    for topic in topic_modeling(processed_text):
        st.write(f"🔹 **{topic[0]}:** {topic[1]}")
    st.markdown("</div>", unsafe_allow_html=True)

    # st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    # st.markdown("<div class='section-title'>📝 Summary</div>", unsafe_allow_html=True)
    # # summary_text = preprocess_for_summary(raw_text)
    # if apply_lemma:
    #  summary_text = preprocess_for_summary_lemmatized(raw_text)
    # else:
    #  summary_text = preprocess_for_summary(raw_text)



    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Keyword Dashboard</div>", unsafe_allow_html=True)

    keywords = get_top_keywords(processed_text, top_n=10)

    if keywords:
     labels, values = zip(*keywords)

     fig, ax = plt.subplots(figsize=(10, 5))
     colors = [
       "#6366f1", "#8b5cf6", "#ec4899", "#f59e0b",
       "#10b981", "#22c55e", "#0ea5e9", "#14b8a6",
       "#f97316", "#a855f7"
    ]
     

     ax.bar(labels, values, color=colors[:len(labels)])
     ax.set_ylabel("Frequency")
     ax.set_xlabel("Keywords")
     ax.set_title("Top Unique Keywords")
     plt.xticks(rotation=45)


     ax.margins(x=0.02, y=0.05)
     plt.tight_layout()

     st.pyplot(fig, use_container_width=True)
    else:
     st.info("Not enough data to generate keyword graph.")

    st.markdown("</div>", unsafe_allow_html=True)



    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📝 Summary</div>", unsafe_allow_html=True)
    # summary_text = preprocess_for_summary(raw_text)
    if apply_lemma:
     summary_text = preprocess_for_summary_lemmatized(raw_text)
    else:
     summary_text = preprocess_for_summary(raw_text)


    # st.info(summarize_text(processed_text))
    st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        padding: 18px 22px;
        border-radius: 14px;
        font-size: 16px;
        line-height: 1.6;
        color: white;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
    ">
        {summarize_text(summary_text)}
    </div>
    """,
    unsafe_allow_html=True

)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>☁️ Word Cloud</div>", unsafe_allow_html=True)
    st.pyplot(generate_wordcloud(processed_text), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


if st.button("🧹 Clear All", use_container_width=True):
    st.rerun()


