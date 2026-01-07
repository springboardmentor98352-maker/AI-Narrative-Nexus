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

# ========================
# Custom CSS for Dark Theme UI
# ========================

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

# Session State (added new keys)
for k in ['text_data', 'cleaned_text', 'tokens', 'cleaned_documents', 'topics_data', 'topic_prevalence', 'model', 'vectorizer', 'sentence_sentiments']:
    if k not in st.session_state:
        if k in ['tokens', 'cleaned_documents', 'sentence_sentiments']:
            st.session_state[k] = []
        elif k in ['topics_data', 'topic_prevalence', 'model', 'vectorizer']:
            st.session_state[k] = None
        else:
            st.session_state[k] = ""

# Input Tabs (unchanged)

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
                        top_keywords = ', '.join(words[:5])  # Top 5 keywords for brevity
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
