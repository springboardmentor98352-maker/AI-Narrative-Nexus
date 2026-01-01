import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px   # visualizations
import pandas as pd           # CSV
import docx                   # DOCX


# ================== PAGE + THEME ==================
st.set_page_config(page_title="NarrativeNexus", layout="wide")

st.markdown("""
<style>

body, [data-testid="stAppViewContainer"] {
    background-color: white !important;
    color: black !important;
}

/* card box for results – disabled (no big boxes) */
.section {
    background: transparent;
    border: none;
    padding: 0;
    margin-top: 18px;
}

/* Section Titles (navy oval) */
.section-title {
    background: navy;
    color: white;
    width: max-content;
    padding: 6px 20px;
    border-radius: 30px;
    font-size: 20px;
    font-weight: 700;
    margin: 10px 0;
}

/* TABS – baby pink */
.stTabs [data-baseweb="tab-list"] {
    background-color: pink;
    border-radius: 12px;
    padding: 10px;
}

/* individual tab text – bigger + more padding */
.stTabs [data-baseweb="tab"] {
    color: navy !important;
    font-weight: 700;
    font-size: 18px;
    padding-top: 10px;
    padding-bottom: 10px;
    padding-left: 22px;
    padding-right: 22px;
}

/* active tab */
.stTabs [aria-selected="true"] {
    background-color: white;
    border-radius: 18px;
    font-weight: 800;
    border: 2px solid navy;
}

</style>
""", unsafe_allow_html=True)

# Stylish single title line
st.markdown("""
<div style="
    text-align:center;
    font-size:42px;
    font-weight:900;
    color:navy;
    font-family:'Trebuchet MS','Segoe UI',sans-serif;
    letter-spacing:1px;
    margin-bottom:25px;">
    <span style="text-transform:uppercase;">NarrativeNexus</span>
    &nbsp; | &nbsp;
    <span>Dynamic Text Analysis Platform</span>
</div>
""", unsafe_allow_html=True)


# ================== NLP SETUP ==================
def setup_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

setup_nltk()


@st.cache_resource
def load_summarizer():
    # heavy model – load once & reuse
    return pipeline("summarization", model="facebook/bart-large-cnn")


summarizer = load_summarizer()


# ================== FUNCTIONS ==================
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens), tokens


def analyze_sentiment(text):
    blob = TextBlob(text).sentiment.polarity
    if blob > 0:
        label = "💚 Positive"
    elif blob < 0:
        label = "❤️ Negative"
    else:
        label = "💛 Neutral"
    return label, blob


def extractive_summary(text):
    sents = nltk.sent_tokenize(text)
    if len(sents) <= 3:
        return text
    freq = {}
    for word in word_tokenize(text.lower()):
        if word.isalnum():
            freq[word] = freq.get(word, 0) + 1
    scores = {}
    for s in sents:
        scores[s] = sum(freq.get(w.lower(), 0) for w in word_tokenize(s) if w.isalnum())
    top = sorted(scores, key=scores.get, reverse=True)[:3]
    return " ".join(top)


# SAFE abstractive summary (paragraph style)
def abstractive_summary(text):
    if not text:
        return ""
    if len(text.split()) < 30:
        return text

    words = text.split()
    if len(words) > 400:
        text_chunk = " ".join(words[:400])
    else:
        text_chunk = text

    try:
        t = summarizer(
            text_chunk,
            max_length=150,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        if t and isinstance(t, list) and "summary_text" in t[0]:
            # ensure proper paragraph spacing
            summary = t[0]["summary_text"].strip()
            return summary
        else:
            return "Abstractive summary not available (unexpected model output)."
    except Exception:
        return "Abstractive summary not available for this input. Try using a shorter text."


def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def cosine_recommendations(query, corpus, top_k=5):
    documents = [query] + corpus
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(documents)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    ranked_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    results = [(corpus[i], float(sims[i])) for i in ranked_idx]
    return results


# ✅ NEW: Overall paragraph-style report
def build_overview_report(text, sentiment_label, sentiment_score, ext_sum, abs_sum, tokens):
    total_words = len(text.split())
    unique_terms = len(set(tokens)) if tokens else 0

    if sentiment_score > 0.1:
        tone_text = "overall positive"
    elif sentiment_score < -0.1:
        tone_text = "overall negative"
    else:
        tone_text = "mostly neutral"

    clean_sentiment = sentiment_label.split(" ", 1)[-1]  # remove emoji

    report = f"""
This text contains approximately {total_words} words with around {unique_terms} important unique terms after preprocessing.
The overall sentiment of the text is {tone_text} ({clean_sentiment}, polarity score {sentiment_score:.2f}).

A brief extractive summary based directly on the most important sentences from the text is:
“{ext_sum}”

An abstractive summary, which rephrases the content in a more natural way, is:
“{abs_sum}”

In simple terms, the text mainly talks about the ideas captured above, with a tone that can be considered {tone_text}.
"""
    return report.strip()


# ================== TABS ==================
tab_overview, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview Report", "Tokenization", "Preprocessing", "Summarisation", "Sentiment Analysis", "Cosine Similarity"]
)


# ================== INPUT UNDER TABS (UPLOAD + TEXT) ==================
with st.container():
    st.markdown('<div class="section-title">📥 Enter Text</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a file (.txt, .csv, .docx) or paste text below 👇",
        type=["txt", "csv", "docx"]
    )

    text_data = ""   # final text used for analysis

    # ---- If a file is uploaded ----
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()

        # TXT
        if file_type == "txt":
            raw_bytes = uploaded_file.read()
            try:
                text_data = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text_data = raw_bytes.decode("latin-1")

        # CSV
        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            text_data = df.to_string()

        # DOCX
        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            paras = [p.text for p in doc.paragraphs if p.text.strip() != ""]
            text_data = "\n".join(paras)

        st.success(f"{uploaded_file.name} uploaded successfully! Text extracted for analysis.")

    # ---- Manual text area (fallback / alternative) ----
    manual_text = st.text_area("Or paste your text here:", height=200)

    if not text_data:
        text_data = manual_text


# ================== PROCESSING + TABS CONTENT ==================
if text_data:

    # Common processing
    pre, tokens = preprocess_text(text_data)
    sentiment_label, sentiment_score = analyze_sentiment(text_data)
    ext_sum = extractive_summary(text_data)
    abs_sum = abstractive_summary(text_data)

    # -------- OVERVIEW REPORT (paragraph output) --------
    with tab_overview:
        st.markdown('<div class="section-title">📚 Overall Narrative Report</div>', unsafe_allow_html=True)
        report_text = build_overview_report(
            text=text_data,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            ext_sum=ext_sum,
            abs_sum=abs_sum,
            tokens=tokens
        )
        st.write(report_text)

    # ---------- TAB 1: TOKENIZATION ----------
    with tab1:
        st.markdown('<div class="section-title">📄 Original Text</div>', unsafe_allow_html=True)
        st.write(text_data)

        st.markdown('<div class="section-title">🧩 Tokens</div>', unsafe_allow_html=True)
        st.write(tokens)
        st.caption(f"Total tokens after basic cleaning: {len(tokens)}")

        if tokens:
            freq = Counter(tokens)
            top_tokens = freq.most_common(15)
            words = [w for w, _ in top_tokens]
            counts = [c for _, c in top_tokens]

            st.markdown('<div class="section-title">📊 Top 15 Tokens</div>', unsafe_allow_html=True)
            fig_tok = px.bar(
                x=words,
                y=counts,
                labels={"x": "Token", "y": "Frequency"},
                title="Top 15 Tokens"
            )
            st.plotly_chart(fig_tok, use_container_width=True)

    # ---------- TAB 2: PREPROCESSING ----------
    with tab2:
        st.markdown('<div class="section-title">✨ Preprocessed Text</div>', unsafe_allow_html=True)
        st.write(pre)

        st.markdown('<div class="section-title">🌈 Word Cloud</div>', unsafe_allow_html=True)
        st.pyplot(generate_wordcloud(pre))

        if tokens:
            token_lengths = [len(t) for t in tokens]
            st.markdown('<div class="section-title">📉 Token Length Distribution</div>', unsafe_allow_html=True)
            fig_len = px.histogram(
                x=token_lengths,
                nbins=10,
                labels={"x": "Token Length (characters)", "y": "Count"},
                title="Distribution of Token Lengths"
            )
            st.plotly_chart(fig_len, use_container_width=True)

    # ---------- TAB 3: SUMMARISATION ----------
    with tab3:
        st.markdown('<div class="section-title">🧾 Extractive Summary</div>', unsafe_allow_html=True)
        st.write(ext_sum)

        st.markdown('<div class="section-title">🧠 Abstractive Summary</div>', unsafe_allow_html=True)
        st.write(abs_sum)

        original_len = len(text_data.split())
        ext_len = len(ext_sum.split())
        abs_len = len(abs_sum.split())

        st.markdown('<div class="section-title">📊 Word Count Comparison</div>', unsafe_allow_html=True)
        labels_wc = ["Original", "Extractive", "Abstractive"]
        values_wc = [original_len, ext_len, abs_len]
        fig_wc = px.bar(
            x=labels_wc,
            y=values_wc,
            labels={"x": "Text Type", "y": "Word Count"},
            title="Word Count: Original vs Summaries"
        )
        st.plotly_chart(fig_wc, use_container_width=True)

    # ---------- TAB 4: SENTIMENT ----------
    with tab4:
        st.markdown('<div class="section-title">❤️ Sentiment Analysis</div>', unsafe_allow_html=True)
        st.write(sentiment_label)

        st.markdown('<div class="section-title">📊 Sentiment Score</div>', unsafe_allow_html=True)
        fig_sent = px.bar(
            x=["Polarity"],
            y=[sentiment_score],
            labels={"x": "Metric", "y": "Score (-1 to 1)"},
            range_y=[-1, 1],
            title="Overall Sentiment Polarity"
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # ---------- TAB 5: COSINE SIMILARITY ----------
    with tab5:
        st.markdown('<div class="section-title">🔗 Cosine Similarity – Semantic Recommender</div>', unsafe_allow_html=True)

        st.write("We compare your input text with a small sample corpus using TF-IDF + cosine similarity.")
        st.write("You can replace this sample corpus with your own dataset later.")

        sample_corpus = [
            "Relaxing smooth jazz songs perfect for late night listening.",
            "High energy workout music with strong beats and fast tempo.",
            "Romantic Bollywood love songs with emotional lyrics.",
            "Lo-fi chill beats to help you study and focus.",
            "Classic rock songs with powerful guitar solos."
        ]

        top_k = st.slider("Number of recommendations:", min_value=1, max_value=5, value=3)

        results = cosine_recommendations(text_data, sample_corpus, top_k=top_k)

        st.subheader("Top Similar Results")
        labels_sim = []
        scores_sim = []

        for i, (doc, score) in enumerate(results, start=1):
            st.markdown(f"**{i}.** {doc}")
            st.caption(f"Similarity score: `{score:.3f}`")
            labels_sim.append(f"Result {i}")
            scores_sim.append(score)

        if scores_sim:
            st.markdown('<div class="section-title">📊 Similarity Scores</div>', unsafe_allow_html=True)
            fig_sim = px.bar(
                x=labels_sim,
                y=scores_sim,
                labels={"x": "Result", "y": "Cosine Similarity"},
                title="Cosine Similarity with Input Text"
            )
            st.plotly_chart(fig_sim, use_container_width=True)

else:
    st.info("Please enter some text above or upload a file to start the analysis.")
