import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
import gensim
from gensim import corpora
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter


# ================== PAGE + THEME ==================
st.set_page_config(page_title="NarrativeNexus", page_icon="✨", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #E3FDFD, #CBF1F5, #A6E3E9);
}
.section {
    background: white;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}
.section-title {
    font-size: 18px;
    font-weight: bold;
    color: BLUE;
    margin-bottom: 5px;
}
.stButton > button, .stDownloadButton > button {
    background: #0077b6;
    color: 7;
    border: none;
    border-radius: 8px;
    padding: 6px 10px;
}
textarea {
    background-color:  #999999 !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ================== NLTK/MODELS ==================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load('en_core_web_sm')

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# ================== FUNCTIONS ==================
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens), tokens


def lda_topics(texts, num_topics=5):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]
    if len(dictionary) == 0:
        return ["Not enough data"], None, None

    lda_model = gensim.models.LdaModel(
        corpus,
        num_topics=min(num_topics, len(dictionary)),
        id2word=dictionary,
        passes=10
    )
    raw = lda_model.print_topics(num_words=5)
    return [f"Topic {i}: {w}" for i, w in raw], lda_model, corpus


def nmf_topics(texts, num_topics=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    if tfidf.shape[1] == 0:
        return ["Not enough data"]

    nmf = NMF(n_components=min(num_topics, tfidf.shape[1]), random_state=42)
    nmf.fit(tfidf)

    names = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(nmf.components_):
        words = [names[i] for i in topic.argsort()[:-6:-1]]
        topics.append(f"Topic {idx}: {' '.join(words)}")
    return topics


def analyze_sentiment(text):
    blob = TextBlob(text).sentiment.polarity
    if blob > 0:
        return "💚 Positive"
    elif blob < 0:
        return "❤️ Negative"
    else:
        return "💛 Neutral"


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


def abstractive_summary(text):
    t = summarizer(text[:3000], max_length=150, min_length=30, do_sample=False)
    return t[0]["summary_text"]


def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def generate_report(pre, lda, nmf, sent, ext, abs_):
    return f"""
NarrativeNexus Analysis Report

Preprocessed:
{pre}

LDA Topics:
{lda}

NMF Topics:
{nmf}

Sentiment:
{sent}

Extractive Summary:
{ext}

Abstractive Summary:
{abs_}
"""


# ================== APP ==================
st.title("✨ NarrativeNexus – Creative AI Text Analysis ✨")

st.sidebar.header("Input")
opt = st.sidebar.selectbox("Choose", ["Paste", "Upload"])
text_data = ""

if opt == "Paste":
    text_data = st.text_area("📝 Paste your text here:")
else:
    file = st.file_uploader("📂 Upload a .txt file", type=["txt"])
    if file:
        text_data = file.read().decode(errors="ignore")


# ================== MAIN ==================
if text_data:

    # ------- Original -------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Original Text</div>', unsafe_allow_html=True)
    st.write(text_data)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------- Preprocessed -------
    pre, tokens = preprocess_text(text_data)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✨ Preprocessed Text</div>', unsafe_allow_html=True)
    st.write(pre)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------- Sentiment -------
    sent = analyze_sentiment(text_data)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">❤️ Sentiment</div>', unsafe_allow_html=True)
    st.write(sent)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------- Topic Models (used in report, can hide in UI if you want) -------
    lda, _, _ = lda_topics([tokens])
    nmf = nmf_topics([pre])

    # ------- Summaries -------
    ext = extractive_summary(text_data)
    abs_ = abstractive_summary(text_data)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 Summaries</div>', unsafe_allow_html=True)
    st.write("**Extractive:**", ext)
    st.write("**Abstractive:**", abs_)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------- Word Cloud -------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌈 Word Cloud</div>', unsafe_allow_html=True)
    st.pyplot(generate_wordcloud(pre))
    st.markdown('</div>', unsafe_allow_html=True)

    # ================== NEW VISUALIZATIONS ==================

    # 📊 Sentiment Visualization
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Sentiment Visualization</div>', unsafe_allow_html=True)

    sent_labels = ["Positive", "Negative", "Neutral"]
    sent_values = [
        1 if "Positive" in sent else 0,
        1 if "Negative" in sent else 0,
        1 if "Neutral" in sent else 0,
    ]
    fig_sent = px.bar(
        x=sent_labels,
        y=sent_values,
        labels={"x": "Sentiment", "y": "Score"},
        title="Sentiment Score"
    )
    st.plotly_chart(fig_sent, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 📈 Top Keywords Visualization
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Top Keywords</div>', unsafe_allow_html=True)

    if tokens:
        freq = Counter(tokens)
        top_words = freq.most_common(10)
        words = [w for w, c in top_words]
        counts = [c for w, c in top_words]

        fig_kw = px.bar(
            x=words,
            y=counts,
            labels={"x": "Keyword", "y": "Frequency"},
            title="Top 10 Keywords"
        )
        st.plotly_chart(fig_kw, use_container_width=True)
    else:
        st.write("Not enough tokens to visualize keywords.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ------- Report Download -------
    if st.button("Download Report"):
        report = generate_report(pre, lda, nmf, sent, ext, abs_)
        st.download_button("Download", report, file_name="report.txt")

# ================== END ==================