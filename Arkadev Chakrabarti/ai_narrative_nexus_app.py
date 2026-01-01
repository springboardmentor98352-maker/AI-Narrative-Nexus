import streamlit as st
import pandas as pd
from docx import Document
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from heapq import nlargest
import numpy as np
import base64
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS


# Transformers for Abstractive Summarization
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# PDF Support
try:
    import PyPDF2
except ImportError:
    st.stop()

# WordCloud Support
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    for r in ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']:
        nltk.download(r, quiet=True)

download_nltk_data()

# ============================
# Custom CSS for Dark Theme UI
# ============================

st.markdown("""
    <style>
    /* Main dark background */
    .stApp {
        background: linear-gradient(to bottom, #0f172a, #1e293b);
        color: #e2e8f0;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #60a5fa !important;
    }
    
    /* General text */
    .css-1d391kg p, .css-1v3fvcr, div.stMarkdown, label {
        color: #cbd5e1 !important;
    }
    
    /* Sidebar dark */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.4);
    }
    
    div.stButton > button[kind="primary"] {
        background-color: #8b5cf6;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #7c3aed;
        box-shadow: 0 6px 12px rgba(124, 58, 237, 0.4);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        color: #60a5fa;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    .streamlit-expanderContent {
        background-color: #111827;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.5rem;
    }
    [data-testid="stMetricLabel"] > label {
        color: #94a3b8 !important;
    }
    [data-testid="stMetricValue"] {
        color: #60a5fa !important;
    }
    
    /* Text inputs and areas */
    .stTextArea textarea, .stTextInput input {
        background-color: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa;
        border-bottom: 2px solid #60a5fa;
    }
    
    /* Alerts */
    .stSuccess { background-color: #166534; color: #dcfce7; }
    .stInfo { background-color: #1e40af; color: #dbeafe; }
    .stWarning { background-color: #854d0e; color: #fef3c7; }
    .stError { background-color: #991b1b; color: #fee2e2; }
    </style>
    """, unsafe_allow_html=True)

# ============
# File Readers
# ============

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

# ======================
# Text Cleaning
# ======================

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

# ======================
# Sentiment Analysis
# ======================

def analyze_sentiment(text):
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    if pol > 0.1: cat = "Positive"
    elif pol < -0.1: cat = "Negative"
    else: cat = "Neutral"
    return {"polarity": pol, "subjectivity": blob.sentiment.subjectivity, "category": cat}

# ===============================================
# Topic Modeling with Document-Topic Distribution
# ===============================================

def perform_topic_modeling(documents, num_topics=5, algorithm='LDA', top_words=10):
    if not documents or len(documents) == 0:
        st.warning("No documents available for topic modeling.")
        return [], None, None

    n_docs = len(documents)

    # Dynamically adjust min_df and max_df
    if n_docs < 10:
        min_df = 1          # Allow words appearing in just 1 document
        max_df = 1.0        # No upper limit
    elif n_docs < 50:
        min_df = 1
        max_df = 0.8
    else:
        min_df = 2
        max_df = 0.95

    try:
        if algorithm == 'LDA':
            vectorizer = CountVectorizer(
                min_df=min_df,
                max_df=max_df,
                stop_words='english'
            )
            dtm = vectorizer.fit_transform(documents)
            model = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=20
            )
        elif algorithm == 'NMF':
            vectorizer = TfidfVectorizer(
                min_df=min_df,
                max_df=max_df,
                stop_words='english'
            )
            dtm = vectorizer.fit_transform(documents)
            model = NMF(
                n_components=num_topics,
                random_state=42,
                init='nndsvd',
                max_iter=300
            )
        else:
            raise ValueError("Unsupported algorithm.")

        # If after filtering, we have no features, then abort
        if dtm.shape[1] == 0:
            st.warning("Not enough unique words after filtering to perform topic modeling. Try with longer text.")
            return [], None, None

        model.fit(dtm)
        doc_topic_dist = model.transform(dtm)
        topic_prevalence = np.mean(doc_topic_dist, axis=0)

        feature_names = vectorizer.get_feature_names_out()

        topics = []
        for topic_idx, topic in enumerate(model.components_):
            sorted_indices = topic.argsort()[::-1]
            top_features = [feature_names[i] for i in sorted_indices[:top_words]]
            top_scores = [topic[i] for i in sorted_indices[:top_words]]
            max_score = top_scores[0] if top_scores else 1
            normalized_scores = [score / max_score if max_score > 0 else 0 for score in top_scores]
            topics.append((topic_idx, top_features, top_scores, normalized_scores))

        return topics, topic_prevalence, vectorizer

    except ValueError as e:
        if "max_df corresponds to < documents than min_df" in str(e):
            st.error("Text is too short or has too few repeated words for topic modeling with current settings.")
            st.info("Tip: Try uploading a longer document (at least 10–15 sentences) for better topic results.")
        else:
            st.error(f"Error during topic modeling: {str(e)}")
        return [], None, None

from collections import Counter
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS


# ========================
# Word Cloud Utilities
# ========================

STOP_WORDS = frozenset(STOPWORDS)


def build_word_frequencies(tokens, min_freq=2):
    """
    Efficiently build word frequencies from large token lists.
    Filters stopwords, short words, non-alpha tokens.
    """

    counter = Counter()

    for token in tokens:
        token = token.lower()

        if (
            token.isalpha()
            and len(token) >= 3
            and token not in STOP_WORDS
        ):
            counter[token] += 1

    # Remove rare/noisy words
    return {
        word: count
        for word, count in counter.items()
        if count >= min_freq
    }


def generate_wordcloud(
    tokens,
    width=900,
    height=450,
    max_words=200,
    background_color="white"
):
    """
    Generates a WordCloud image buffer (PNG) optimized for Streamlit.
    Returns BytesIO or None.
    """

    word_freq = build_word_frequencies(tokens)

    if not word_freq:
        return None

    wc = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color=background_color,
        stopwords=STOP_WORDS,
        collocations=False,
        prefer_horizontal=0.9,
        min_word_length=3,
        normalize_plurals=True
    ).generate_from_frequencies(word_freq)

    # Convert to PIL Image
    pil_img = wc.to_image()

    # Write image to memory buffer
    img_buffer = BytesIO()
    pil_img.save(img_buffer, format="PNG", optimize=True)
    img_buffer.seek(0)

    return img_buffer

# ========================
# Summarization Functions
# ========================

def normalize_topic_words(words):
    """
    Ensures topic words are always a list of strings.
    Handles strings, lists, tuples, numpy arrays, etc.
    """
    if not words:
        return []

    if isinstance(words, str):
        return [words]

    if hasattr(words, "__iter__"):
        # Handle (word, score) tuples or mixed structures
        return [
            w if isinstance(w, str) else str(w[0])
            for w in words
        ]

    return [str(words)]

def build_themes_text(topics_data):
    """
    Formats topic data into readable theme text.
    """
    themes = []

    for i, (words, *_ ) in enumerate(topics_data):
        normalized_words = normalize_topic_words(words)
        theme_line = f"Topic {i + 1}: {', '.join(normalized_words)}"
        themes.append(theme_line)

    return "\n".join(themes)

def prepare_summary_input(themes_text, full_text, max_context_len=1500):
    """
    Combines themes and context into a single summarization input.
    """
    context = full_text[:max_context_len]
    return f"Themes:\n{themes_text}\n\nContext:\n{context}"

def extractive_summarize(text, num_sentences=3):
    if not text.strip():
        return "No text to summarize."
    
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    stop_words = set(stopwords.words("english"))
    word_freq = Counter(word.lower() for word in word_tokenize(text) if word.lower() not in stop_words and word.isalpha())
    
    sentence_scores = {}
    for sent in sentences:
        words = word_tokenize(sent.lower())
        score = sum(word_freq.get(w, 0) for w in words if w not in stop_words and w.isalpha())
        sentence_scores[sent] = score / (len(words) + 1)
    
    top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_sentences = sorted(top_sentences, key=lambda x: text.index(x))
    
    return ' '.join(summary_sentences)

@st.cache_resource
def get_summarizer():
    if pipeline is None:
        raise ImportError("Transformers library not installed.")
    return pipeline("summarization", model="facebook/bart-large-cnn")

def abstractive_summarize(text, max_length=130, min_length=30):
    if not text.strip():
        return "No text to summarize."
    
    try:
        summarizer = get_summarizer()
        if len(text) > 4000:
            text = text[:4000] + "..."
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    except ImportError:
        return "Install 'transformers' and 'torch' for abstractive summarization."
    except Exception as e:
        return f"Error: {str(e)}"

# ========================
# Streamlit App
# ========================

st.set_page_config(page_title="Narrative Nexus", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.8rem; color: #60a5fa; margin-bottom: 0;">📊 Narrative Nexus</h1>
        <p style="font-size: 1.5rem; color: #94a3b8; margin-top: 0.5rem;">Dynamic AI-Powered Text Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=350)
    st.markdown("<h2 style='color:#8b5cf6; text-align:center;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    
    st.markdown("### 📄 Text Cleaning")
    remove_stopwords = st.checkbox("Remove stopwords", True)
    apply_lemmatization = st.checkbox("Apply lemmatization", True)
    min_word_len = st.slider("Min word length", 1, 10, 2)
    top_n_words = st.slider("Top frequent words", 5, 50, 20)
    
    st.markdown("### 🧠 Topic Modeling")
    num_topics = st.slider("Number of topics", 2, 12, 5)
    topic_algorithm = st.selectbox("Algorithm", ["LDA", "NMF"])
    top_words_per_topic = st.slider("Words per topic", 5, 20, 10)
    
    st.markdown("### 📝 Summarization")
    summ_type = st.selectbox("Technique", ["None", "Extractive", "Abstractive"])
    if summ_type == "Extractive":
        num_sents = st.slider("Sentences in summary", 1, 10, 3)
    elif summ_type == "Abstractive":
        min_len = st.slider("Min length", 20, 100, 30)
        max_len = st.slider("Max length", 50, 300, 130)

# Session State
for k in ['text_data', 'cleaned_text', 'tokens', 'cleaned_documents', 'topics_data', 'topic_prevalence', 'model', 'vectorizer', 'sentence_sentiments']:
    if k not in st.session_state:
        if k in ['tokens', 'cleaned_documents', 'sentence_sentiments']:
            st.session_state[k] = []
        elif k in ['topics_data', 'topic_prevalence', 'model', 'vectorizer']:
            st.session_state[k] = None
        else:
            st.session_state[k] = ""

# Input Tabs

tab_upload, tab_manual = st.tabs(["📤 Upload File", "✍️ Enter Text Manually"])

with tab_upload:
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "docx", "pdf"])
    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        with st.spinner(f"Loading {uploaded_file.name}..."):
            if ext == "txt": st.session_state.text_data = read_txt(uploaded_file)
            elif ext == "csv":
                df = read_csv(uploaded_file)
                cols = df.select_dtypes(include='object').columns
                col = st.selectbox("Select text column", cols)
                st.session_state.text_data = " ".join(df[col].dropna().astype(str))
            elif ext == "docx": st.session_state.text_data = read_docx(uploaded_file)
            elif ext == "pdf": st.session_state.text_data = read_pdf(uploaded_file)
        
        st.success("File loaded successfully!")
        st.text_area("Preview", st.session_state.text_data, height=250)

with tab_manual:
    manual_text = st.text_area("Paste your text here", height=350, placeholder="Type or paste your text...")
    if manual_text.strip():
        st.session_state.text_data = manual_text
        st.success("Text entered!")

# Main Analysis
if st.session_state.text_data and st.session_state.text_data.strip():
    st.markdown("---")
    st.header("🔍 Analysis Dashboard")

    col_main, col_side = st.columns([3, 1])

    with col_main:
        if st.button("🧹 Clean & Analyze Text", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                sentences = sent_tokenize(st.session_state.text_data)
                cleaned_docs, all_tokens = [], []
                sentence_cats = []
                for sent in sentences:
                    cleaned_str, tokens = clean_text(sent, remove_stopwords, apply_lemmatization, min_word_len)
                    if cleaned_str:
                        cleaned_docs.append(cleaned_str)
                        all_tokens.extend(tokens)
                        sent_sent = analyze_sentiment(sent)['category']
                        sentence_cats.append(sent_sent)
                
                st.session_state.cleaned_text = " ".join(cleaned_docs)
                st.session_state.tokens = all_tokens
                st.session_state.cleaned_documents = cleaned_docs
                st.session_state.sentence_sentiments = sentence_cats
                st.success("Processing complete!")

                # Statistics
                st.subheader("📊 Basic Statistics")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Tokens", len(all_tokens))
                c2.metric("Unique Tokens", len(set(all_tokens)))
                c3.metric("Sentences", len(sentences))
                c4.metric("Lexical Diversity", f"{len(set(all_tokens))/len(all_tokens):.3f}" if all_tokens else "0")

                # Word Cloud
                wc_img = generate_wordcloud(all_tokens)

                if wc_img:
                     st.subheader("📌 WordCloud Overview")
                     st.image(wc_img, use_container_width=True)
                else:
                    st.warning("Not enough meaningful words to generate WordCloud.")
                
                st.download_button(
                    label="⬇️ Download WordCloud",
                    data=wc_img,
                    file_name="wordcloud.png",
                    mime="image/png"
                )

                # Frequent Words Bar
                st.subheader("🔤 Most Frequent Words")
                common = Counter(all_tokens).most_common(top_n_words)
                if common:
                    df_freq = pd.DataFrame(common, columns=["Word", "Frequency"])
                    fig = px.bar(df_freq, x="Frequency", y="Word", orientation='h',
                                 color="Frequency", color_continuous_scale="Plasma",
                                 text="Frequency")
                    fig.update_layout(height=600, plot_bgcolor='#111827', paper_bgcolor='#111827')
                    st.plotly_chart(fig, use_container_width=True)

                # Sentiment Distribution Bar
                st.subheader("😊 Sentiment Distribution (Sentence-Level)")
                if sentence_cats:
                    sent_count = Counter(sentence_cats)
                    df_sent = pd.DataFrame(list(sent_count.items()), columns=["Sentiment", "Count"])
                    df_sent = df_sent.sort_values("Count", ascending=False)
                    fig_sent = px.bar(df_sent, x="Sentiment", y="Count", color="Sentiment",
                                      color_discrete_map={"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#94a3b8"})
                    fig_sent.update_layout(plot_bgcolor='#111827', paper_bgcolor='#111827')
                    st.plotly_chart(fig_sent, use_container_width=True)

        if st.session_state.cleaned_documents:
            if st.button("🧩 Perform Topic Modeling", type="primary", use_container_width=True):
                with st.spinner("Discovering topics..."):
                    topics_data, prevalence, vectorizer = perform_topic_modeling(
                        st.session_state.cleaned_documents,
                        num_topics=num_topics,
                        algorithm=topic_algorithm,
                        top_words=top_words_per_topic
                    )
                    st.session_state.topics_data = topics_data
                    st.session_state.topic_prevalence = prevalence
                    st.session_state.vectorizer = vectorizer

                    st.subheader(f"🧠 {topic_algorithm} Topics")
                    for idx, words, scores, norm_scores in topics_data:
                        with st.expander(f"Topic {idx+1}: {', '.join(words[:5])}...", expanded=True):
                            df_topic = pd.DataFrame({
                                "Word": words,
                                "Score": [f"{s:.4f}" for s in scores],
                                "Relative": norm_scores
                            })
                            fig = px.bar(df_topic, x="Relative", y="Word", orientation='h',
                                         color="Relative", color_continuous_scale="Viridis",
                                         text="Score")
                            fig.update_layout(height=max(350, 30*len(words)), plot_bgcolor='#111827', paper_bgcolor='#111827')
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(df_topic[["Word", "Score"]], use_container_width=True)

                    # Topic Prevalence Graph
                    if prevalence is not None:
                        st.subheader("📈 Topic Prevalence Across Document")
                        df_prev = pd.DataFrame({
                            "Topic": [f"Topic {i+1}" for i in range(len(prevalence))],
                            "Prevalence (%)": prevalence * 100
                        })
                        fig_prev = px.bar(df_prev, x="Topic", y="Prevalence (%)", color="Topic",
                                          color_discrete_sequence=px.colors.sequential.Viridis)
                        fig_prev.update_layout(plot_bgcolor='#111827', paper_bgcolor='#111827', showlegend=False)
                        st.plotly_chart(fig_prev, use_container_width=True)

                # --- Summary of Themes ---
                summary = None
                themes_text = "No topics generated yet."

                if st.session_state.topics_data:
                    themes_text = build_themes_text(st.session_state.topics_data)

                if summ_type != "None" and st.session_state.topics_data:
                    st.subheader("📜 Summary of Themes")
                    with st.spinner("Generating summary..."):
                        text_to_sum = prepare_summary_input(
                            themes_text,
                            st.session_state.text_data
                        )
                        if summ_type == "Extractive":
                            summary = extractive_summarize(text_to_sum, num_sents)
                        elif summ_type == "Abstractive":
                            summary = abstractive_summarize(text_to_sum, max_len, min_len)
                    if summary:
                        st.info(summary)

                # --- Comprehensive Report ---
                st.subheader("📑 Comprehensive Report")

                from datetime import datetime
                current_timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

                overall_sent = analyze_sentiment(st.session_state.text_data)['category']

                # Default values if topic modeling hasn't been run
                themes_text = "No topics identified yet (run Topic Modeling to see themes)."
                dominant_topics_desc = "Not available — please run Topic Modeling first."

                if st.session_state.topics_data and st.session_state.topic_prevalence is not None:
                    # Build full themes list
                    themes_lines = []
                    for idx, (words, _, _, _) in enumerate(st.session_state.topics_data):
                        word_list = normalize_topic_words(words)
                        themes_lines.append(f"**Topic {idx + 1}:** {', '.join(word_list[:8])}")
                    themes_text = "\n".join(themes_lines)

                    # Get top 3 dominant topics with their representative keywords
                    prevalence = st.session_state.topic_prevalence
                    top_indices = np.argsort(prevalence)[-3:][::-1]  # Highest to lowest

                    dominant_parts = []
                    for rank, topic_idx in enumerate(top_indices, 1):
                        words = normalize_topic_words(st.session_state.topics_data[topic_idx][1])
                        top_keywords = ', '.join(words[:5])  # Top 5 keywords
                        percentage = prevalence[topic_idx] * 100
                        dominant_parts.append(f"{rank}. **Topic {topic_idx + 1}** ({percentage:.1f}% prevalence): {top_keywords}")

                    dominant_topics_desc = "\n".join(dominant_parts)

                # Final report with timestamp and richer content
                report = f"""
**Narrative Nexus – Text Analysis Report**  
*Generated on {current_timestamp}*

**Document Overview**
- Total sentences: {len(sent_tokenize(st.session_state.text_data))}
- Total tokens (after cleaning): {len(st.session_state.tokens) if st.session_state.tokens else 'N/A'}

**Overall Sentiment**  
**{overall_sent}** (Polarity: {analyze_sentiment(st.session_state.text_data)['polarity']:.3f})

**Key Themes Identified**
{themes_text}

**Top 3 Dominant Themes**
{dominant_topics_desc}

**Insights & Actionable Recommendations**
- The primary focus of the text revolves around the dominant themes listed above.
- Sentiment is **{overall_sent.lower()}**, suggesting {'strong positive engagement and approval' if overall_sent == 'Positive' else 'areas of concern that may require attention or improvement' if overall_sent == 'Negative' else 'a balanced or neutral viewpoint with room for deeper interpretation'}.
- {'Prioritize communication, content, or strategy around the top dominant themes to maximize impact and resonance.' if st.session_state.topics_data else 'Run Topic Modeling to unlock specific theme-based recommendations.'}
- For longitudinal tracking, consider analyzing multiple documents over time to monitor shifts in themes and sentiment.
"""

                st.markdown(report)

    with col_side:
        st.subheader("⚡ Quick Tools")
        
        if st.button("😊 Overall Sentiment", use_container_width=True):
            sent = analyze_sentiment(st.session_state.text_data)
            if sent["category"] == "Positive":
                st.success("😊 Positive")
            elif sent["category"] == "Negative":
                st.error("😔 Negative")
            else:
                st.info("😐 Neutral")
            st.metric("Polarity", f"{sent['polarity']:.3f}")
            st.metric("Subjectivity", f"{sent['subjectivity']:.3f}")

        if st.session_state.cleaned_text:
            st.download_button("💾 Download Cleaned Text", st.session_state.cleaned_text,
                               "cleaned_text.txt", use_container_width=True)

st.markdown("---")
st.caption("**Narrative Nexus** — Dynamic Text Analysis Platform | Developed by Arkadev")
