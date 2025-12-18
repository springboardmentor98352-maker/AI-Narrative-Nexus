import streamlit as st
import pandas as pd
from docx import Document
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from heapq import nlargest

# Transformers for Abstractive Summarization
try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # Will handle in the function

# PDF Support
try:
    import PyPDF2
except ImportError:
    st.stop()

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    for r in ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']:
        nltk.download(r, quiet=True)

download_nltk_data()

# ========================
# File Readers
# ========================

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

# ========================
# Text Cleaning
# ========================

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

# ========================
# Sentiment
# ========================

def analyze_sentiment(text):
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    if pol > 0.1: cat = "Positive"
    elif pol < -0.1: cat = "Negative"
    else: cat = "Neutral"
    return {"polarity": pol, "subjectivity": blob.sentiment.subjectivity, "category": cat}

# ========================
# Topic Modeling with Importance Scores
# ========================

def perform_topic_modeling(documents, num_topics=5, algorithm='LDA', top_words=10):
    if not documents:
        return []

    if algorithm == 'LDA':
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(documents)
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    elif algorithm == 'NMF':
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(documents)
        model = NMF(n_components=num_topics, random_state=42, init='nndsvd')  # Better initialization for stability
    else:
        raise ValueError("Unsupported algorithm. Choose 'LDA' or 'NMF'.")

    model.fit(dtm)
    
    feature_names = vectorizer.get_feature_names_out()
    
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        # Get indices sorted by importance descending
        sorted_indices = topic.argsort()[::-1]
        top_features = [feature_names[i] for i in sorted_indices[:top_words]]
        top_scores = [topic[i] for i in sorted_indices[:top_words]]
        
        # Normalize scores for better visualization (max = 1)
        max_score = top_scores[0] if top_scores else 1
        normalized_scores = [score / max_score for score in top_scores]
        
        topics.append((topic_idx, top_features, top_scores, normalized_scores))
    
    return topics, vectorizer, model

# ========================
# Summarization Functions
# ========================

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
        words = word_tokenize(sent)
        score = sum(word_freq.get(word.lower(), 0) for word in words if word.lower() not in stop_words and word.isalpha())
        sentence_scores[sent] = score / len(words) if words else 0
    
    top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # Sort top sentences by their order in the original text
    summary_sentences = sorted(top_sentences, key=lambda x: text.index(x))
    
    return ' '.join(summary_sentences)

@st.cache_resource
def get_summarizer():
    if pipeline is None:
        raise ImportError("Transformers library not installed.")
    return pipeline("summarization", model="facebook/bart-large-cnn")  # More reliable default model

def abstractive_summarize(text, max_length=130, min_length=30):
    if not text.strip():
        return "No text to summarize."
    
    try:
        summarizer = get_summarizer()
        if len(text) > 4000:
            text = text[:4000] + "... (truncated for summarization)"
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    except ImportError:
        return "Please install the 'transformers' library to use abstractive summarization. Run 'pip install transformers torch'."
    except Exception as e:
        return f"Error in abstractive summarization: {str(e)}"

# ========================
# Streamlit App
# ========================

st.set_page_config(page_title="Narrative Nexus", layout="wide")
st.title("Narrative Nexus")
st.write("Dynamic Text Analysis Platform")

# Sidebar
with st.sidebar:
    st.header("Settings & Configuration")
    st.subheader("Text Cleaning Options")
    remove_stopwords = st.checkbox("Remove stopwords", True)
    apply_lemmatization = st.checkbox("Apply lemmatization", True)
    min_word_len = st.slider("Min word length", 1, 10, 2)
    top_n_words = st.slider("Top N frequent words to show", 5, 50, 20)
    
    st.subheader("Topic Modeling Options")
    num_topics = st.slider("Number of topics", 2, 10, 5)
    topic_algorithm = st.selectbox("Algorithm", ["LDA", "NMF"])
    top_words_per_topic = st.slider("Top words per topic", 5, 20, 10)
    
    st.subheader("Summarization Options")
    summ_type = st.selectbox("Summarization Technique", ["None", "Extractive", "Abstractive"])
    if summ_type == "Extractive":
        num_sents = st.slider("Number of sentences in summary", 1, 10, 3)
    elif summ_type == "Abstractive":
        min_len = st.slider("Min summary length", 10, 100, 30)
        max_len = st.slider("Max summary length", 50, 500, 130)

# Session State
for k in ['text_data', 'cleaned_text', 'tokens', 'cleaned_documents', 'topics_data']:
    if k not in st.session_state:
        if k == 'topics_data':
            st.session_state[k] = None
        elif k in ['tokens', 'cleaned_documents']:
            st.session_state[k] = []
        else:
            st.session_state[k] = ""

tab1, tab2 = st.tabs(["Upload File", "Enter Text Manually"])

# Upload
with tab1:
    file = st.file_uploader("Upload file", type=["txt","csv","docx","pdf"])
    if file:
        ext = file.name.split(".")[-1].lower()
        st.success(f"Uploaded: {file.name}")

        if ext == "txt": st.session_state.text_data = read_txt(file)
        elif ext == "csv":
            df = read_csv(file)
            st.dataframe(df)
            cols = df.select_dtypes(include='object').columns
            if len(cols)>0:
                col = st.selectbox("Text column", cols)
                st.session_state.text_data = " ".join(df[col].dropna().astype(str))
        elif ext == "docx": st.session_state.text_data = read_docx(file)
        elif ext == "pdf":
            with st.spinner("Extracting PDF..."):
                st.session_state.text_data = read_pdf(file)

        if st.session_state.text_data:
            st.text_area("Extracted Text", st.session_state.text_data, height=300)

# Manual
with tab2:
    txt = st.text_area("Enter text here manually:", height=300)
    if txt.strip():
        st.session_state.text_data = txt
        st.success("Text loaded!")

# ========================
# ANALYSIS
# ========================

if st.session_state.text_data and st.session_state.text_data.strip():
    st.markdown("---")
    st.header("Text Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Clean & Analyze Text", type="primary", use_container_width=True):
            with st.spinner("Processing text..."):
                sentences = sent_tokenize(st.session_state.text_data)
                cleaned_docs = []
                all_tokens = []
                for sent in sentences:
                    cleaned_str, tokens = clean_text(
                        sent,
                        remove_stopwords,
                        apply_lemmatization,
                        min_word_len
                    )
                    if cleaned_str:
                        cleaned_docs.append(cleaned_str)
                        all_tokens.extend(tokens)
                
                st.session_state.cleaned_text = " ".join(cleaned_docs)
                st.session_state.tokens = all_tokens
                st.session_state.cleaned_documents = cleaned_docs

                st.success("Analysis Complete!")

                # Token Stats
                total = len(all_tokens)
                unique = len(set(all_tokens))

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Tokens", total)
                c2.metric("Unique Tokens", unique)
                c3.metric("Lexical Diversity", f"{unique/total:.3f}" if total else "0")

                with st.expander("View Tokens (first 100)"):
                    st.write(all_tokens[:100])

                words_per_sent = [len(word_tokenize(s)) for s in sentences]
                avg_words = sum(words_per_sent) / len(words_per_sent) if words_per_sent else 0

                s1, s2, s3 = st.columns(3)
                s1.metric("Total Sentences", len(sentences))
                s2.metric("Avg Words/Sentence", f"{avg_words:.1f}")
                s3.metric("Longest Sentence", max(words_per_sent) if words_per_sent else 0)

                # MOST FREQUENT WORDS
                st.subheader(f"Top {top_n_words} Most Frequent Words")
                word_counts = Counter(all_tokens)
                common = word_counts.most_common(top_n_words)

                if common:
                    df_freq = pd.DataFrame(common, columns=["Word", "Frequency"])

                    fig = px.bar(
                        df_freq, x="Frequency", y="Word", orientation='h',
                        text="Frequency", color="Frequency",
                        color_continuous_scale="Viridis",
                        title=f"Top {top_n_words} Words"
                    )
                    fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("View & Download Frequency Table"):
                        st.dataframe(df_freq, use_container_width=True)
                        csv = df_freq.to_csv(index=False).encode()
                        st.download_button(
                            "Download CSV",
                            csv,
                            "word_frequency.csv",
                            "text/csv"
                        )
                else:
                    st.info("No tokens after cleaning.")

        # Topic Modeling Section
        if 'cleaned_documents' in st.session_state and st.session_state.cleaned_documents:
            if st.button("Perform Topic Modeling", type="primary", use_container_width=True):
                with st.spinner("Running topic modeling..."):
                    try:
                        topics_data, vectorizer, model = perform_topic_modeling(
                            st.session_state.cleaned_documents,
                            num_topics=num_topics,
                            algorithm=topic_algorithm,
                            top_words=top_words_per_topic
                        )
                        st.session_state.topics_data = topics_data
                        
                        if not topics_data:
                            st.warning("Not enough data for topic modeling.")
                        else:
                            st.subheader(f"{topic_algorithm} Topics (Top {top_words_per_topic} Words)")
                            
                            for topic_idx, top_features, top_scores, normalized_scores in topics_data:
                                st.write(f"**Topic {topic_idx + 1}**")
                                
                                df_topic = pd.DataFrame({
                                    "Word": top_features,
                                    "Importance Score": [f"{score:.4f}" for score in top_scores],
                                    "Relative Importance": normalized_scores
                                })
                                
                                fig_topic = px.bar(
                                    df_topic, 
                                    x="Relative Importance", 
                                    y="Word", 
                                    orientation='h',
                                    text="Importance Score",
                                    color="Relative Importance",
                                    color_continuous_scale="Blues",
                                    title=f"Topic {topic_idx + 1} Word Importance"
                                )
                                fig_topic.update_layout(height=max(300, 30 * len(top_features)), yaxis={'categoryorder':'total ascending'})
                                fig_topic.update_traces(textposition='outside')
                                st.plotly_chart(fig_topic, use_container_width=True)
                                
                                # Also show as table
                                with st.expander(f"View raw scores for Topic {topic_idx + 1}"):
                                    st.dataframe(df_topic[["Word", "Importance Score"]], use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in topic modeling: {str(e)}. Try adjusting parameters or ensuring sufficient text data.")

                # Summarization of Themes and Insights
                if summ_type != "None" and st.session_state.topics_data:
                    with st.spinner("Summarizing themes..."):
                        themes_text = "Identified themes and insights from the text analysis:\n"
                        for topic_idx, top_features, _, _ in st.session_state.topics_data:
                            themes_text += f"Topic {topic_idx + 1}: {', '.join(top_features)}\n"
                        
                        context_text = st.session_state.text_data[:2000]
                        full_text_to_summarize = themes_text + "\n\nSample original text: " + context_text
                        
                        if summ_type == "Extractive":
                            summary = extractive_summarize(full_text_to_summarize, num_sentences=num_sents)
                        elif summ_type == "Abstractive":
                            summary = abstractive_summarize(full_text_to_summarize, max_length=max_len, min_length=min_len)
                        
                        st.subheader("Summary of Themes and Insights")
                        st.write(summary)

    with col2:
        st.subheader("Additional Tools")
        if st.button("Sentiment Analysis", type="secondary", use_container_width=True):
            sent = analyze_sentiment(st.session_state.text_data)
            if sent['category'] == "Positive":
                st.success(f"Positive")
            elif sent['category'] == "Negative":
                st.error(f"Negative")
            else:
                st.info(f"Neutral")

            st.metric("Polarity", f"{sent['polarity']:.3f}")
            st.metric("Subjectivity", f"{sent['subjectivity']:.3f}")

        st.markdown("---")
        if st.session_state.cleaned_text:
            st.download_button(
                "Download Cleaned Text",
                st.session_state.cleaned_text,
                "cleaned_text.txt",
                "text/plain"
            )

st.markdown("---")
st.markdown("**Narrative Nexus** — Dynamic Text Analysis Tool | Developed by Arkadev" )
