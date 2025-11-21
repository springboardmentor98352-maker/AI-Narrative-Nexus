import streamlit as st
import pandas as pd
from docx import Document
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    resources = ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

download_nltk_data()

# File reading functions
def read_txt(file):
    """Read .txt file content."""
    return file.read().decode("utf-8")

def read_csv(file):
    """Read .csv file content and return as DataFrame."""
    return pd.read_csv(file)

def read_docx(file):
    """Read .docx file content."""
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Enhanced text cleaning function
def clean_text(text, remove_stopwords=True, lemmatize=True, min_word_length=2):
    """
    Enhanced text cleaning with configurable options.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Filter by minimum word length
    tokens = [word for word in tokens if len(word) >= min_word_length]
    
    # Lemmatization if requested
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join cleaned tokens
    cleaned_text = " ".join(tokens)
    return cleaned_text

def perform_lda(texts, n_topics=5, n_words=10):
    """
    Perform Latent Dirichlet Allocation for topic modeling.
    """
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Train LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='online'
    )
    lda_output = lda_model.fit_transform(doc_term_matrix)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract topics
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_num': topic_idx + 1,
            'words': top_words,
            'weights': [topic[i] for i in top_words_idx]
        })
    
    return lda_model, lda_output, topics, vectorizer

def perform_nmf(texts, n_topics=5, n_words=10):
    """
    Perform Non-negative Matrix Factorization for topic modeling.
    """
    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # Train NMF model
    nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=200)
    nmf_output = nmf_model.fit_transform(tfidf_matrix)
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Extract topics
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_num': topic_idx + 1,
            'words': top_words,
            'weights': [topic[i] for i in top_words_idx]
        })
    
    return nmf_model, nmf_output, topics, tfidf_vectorizer

def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob.
    Returns polarity (-1 to 1) and subjectivity (0 to 1).
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Categorize sentiment
    if polarity > 0.1:
        category = "Positive"
    elif polarity < -0.1:
        category = "Negative"
    else:
        category = "Neutral"
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'category': category
    }

def analyze_topic_sentiment(texts, topic_assignments):
    """
    Analyze sentiment for each topic.
    """
    topic_sentiments = {}
    
    for idx, text in enumerate(texts):
        # Get dominant topic for this document
        dominant_topic = np.argmax(topic_assignments[idx]) + 1
        
        # Analyze sentiment
        sentiment = analyze_sentiment(text)
        
        if dominant_topic not in topic_sentiments:
            topic_sentiments[dominant_topic] = []
        
        topic_sentiments[dominant_topic].append(sentiment)
    
    # Calculate average sentiment per topic
    topic_summary = {}
    for topic, sentiments in topic_sentiments.items():
        avg_polarity = np.mean([s['polarity'] for s in sentiments])
        avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments])
        
        if avg_polarity > 0.1:
            category = "Positive"
        elif avg_polarity < -0.1:
            category = "Negative"
        else:
            category = "Neutral"
        
        topic_summary[topic] = {
            'avg_polarity': avg_polarity,
            'avg_subjectivity': avg_subjectivity,
            'category': category,
            'doc_count': len(sentiments)
        }
    
    return topic_summary

# Streamlit UI
st.set_page_config(page_title="Narrative Nexus", layout="wide")

st.title("Narrative Nexus")
st.write("Advanced text analysis with topic modeling and sentiment analysis")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Text Cleaning Options")
    remove_stopwords = st.checkbox("Remove stopwords", value=True)
    apply_lemmatization = st.checkbox("Apply lemmatization", value=True)
    min_word_len = st.slider("Minimum word length", 1, 5, 2)
    
    st.subheader("Topic Modeling Options")
    n_topics = st.slider("Number of topics", 2, 10, 5)
    n_top_words = st.slider("Top words per topic", 5, 15, 10)
    algorithm = st.selectbox("Algorithm", ["LDA", "NMF", "Both"])

# Initialize session state
if 'text_data' not in st.session_state:
    st.session_state.text_data = ""
if 'cleaned_text' not in st.session_state:
    st.session_state.cleaned_text = ""

# Main content
tab1, tab2 = st.tabs(["Upload File", "Enter Text Manually"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload your text file", 
        type=["txt", "csv", "docx"], 
        help="Supported formats: .txt, .csv, .docx"
    )

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()
        st.success(f"File uploaded: `{uploaded_file.name}`")

        if file_type == "txt":
            st.session_state.text_data = read_txt(uploaded_file)
            st.text_area("File Content", st.session_state.text_data, height=300, key="txt_preview")
        
        elif file_type == "csv":
            df = read_csv(uploaded_file)
            st.dataframe(df, use_container_width=True)
            
            # If CSV has text columns, allow user to select
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            if text_columns:
                selected_column = st.selectbox("Select text column for analysis", text_columns)
                st.session_state.text_data = " ".join(df[selected_column].astype(str).tolist())
        
        elif file_type == "docx":
            st.session_state.text_data = read_docx(uploaded_file)
            st.text_area("Extracted Text", st.session_state.text_data, height=300, key="docx_preview")

with tab2:
    st.write("Enter your text manually below:")
    user_input = st.text_area("Your Text", height=300, placeholder="Type or paste your text here...", key="manual_input")
    if user_input:
        st.session_state.text_data = user_input
        st.success("Text entered successfully!")

# Text Processing Section
if st.session_state.text_data:
    st.markdown("---")
    st.header("Text Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clean & Tokenize Text", type="primary"):
            with st.spinner("Cleaning text..."):
                st.session_state.cleaned_text = clean_text(
                    st.session_state.text_data,
                    remove_stopwords=remove_stopwords,
                    lemmatize=apply_lemmatization,
                    min_word_length=min_word_len
                )
                
                st.success("Text cleaned successfully!")
                
                # Display cleaned text
                st.subheader("Cleaned Text")
                st.text_area("", st.session_state.cleaned_text, height=200, key="cleaned_preview")
                
                # Tokenization stats
                tokens = st.session_state.cleaned_text.split()
                st.metric("Total Tokens", len(tokens))
                st.metric("Unique Tokens", len(set(tokens)))
                
                # Show token preview
                with st.expander("View Token Preview (First 100 tokens)"):
                    st.write(tokens[:100])
    
    with col2:
        if st.button("Analyze Sentiment", type="primary"):
            if st.session_state.text_data:
                with st.spinner("Analyzing sentiment..."):
                    sentiment = analyze_sentiment(st.session_state.text_data)
                    
                    st.subheader("Sentiment Analysis")
                    
                    # Display sentiment category with color
                    if sentiment['category'] == "Positive":
                        st.success(f"**Sentiment: {sentiment['category']}**")
                    elif sentiment['category'] == "Negative":
                        st.error(f"**Sentiment: {sentiment['category']}**")
                    else:
                        st.info(f"**Sentiment: {sentiment['category']}**")
                    
                    # Display metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Polarity", f"{sentiment['polarity']:.3f}", 
                                 help="Range: -1 (negative) to +1 (positive)")
                    with col_b:
                        st.metric("Subjectivity", f"{sentiment['subjectivity']:.3f}",
                                 help="Range: 0 (objective) to 1 (subjective)")
    
    # Topic Modeling Section
    if st.session_state.cleaned_text:
        st.markdown("---")
        st.header("Topic Modeling")
        
        if st.button("Run Topic Modeling", type="primary"):
            # Split text into sentences for better topic modeling
            sentences = sent_tokenize(st.session_state.text_data)
            
            # Clean each sentence
            cleaned_sentences = [
                clean_text(sent, remove_stopwords, apply_lemmatization, min_word_len) 
                for sent in sentences
            ]
            cleaned_sentences = [s for s in cleaned_sentences if s.strip()]
            
            # Adjust minimum requirement based on available content
            min_required = min(2, max(1, n_topics - 2))
            
            if len(cleaned_sentences) < min_required:
                st.warning(f"Text is too short for meaningful topic modeling. Need at least {min_required} sentences with content.")
            else:
                # If we have fewer sentences than topics, adjust n_topics
                adjusted_n_topics = min(n_topics, max(2, len(cleaned_sentences) - 1))
                if adjusted_n_topics < n_topics:
                    st.info(f"Adjusting number of topics to {adjusted_n_topics} based on document length.")
                with st.spinner("Training topic models..."):
                        # Perform LDA
                        if algorithm in ["LDA", "Both"]:
                            st.subheader("LDA (Latent Dirichlet Allocation)")
                            try:
                                lda_model, lda_output, lda_topics, lda_vectorizer = perform_lda(
                                    cleaned_sentences, adjusted_n_topics, n_top_words
                                )
                                
                                # Display topics
                                for topic in lda_topics:
                                    with st.expander(f"Topic {topic['topic_num']}"):
                                        st.write("**Top Words:**")
                                        words_df = pd.DataFrame({
                                            'Word': topic['words'],
                                            'Weight': topic['weights']
                                        })
                                        st.dataframe(words_df, use_container_width=True)
                                
                                # Analyze sentiment by topic
                                st.subheader("Sentiment Analysis by Topic (LDA)")
                                topic_sentiments = analyze_topic_sentiment(cleaned_sentences, lda_output)
                                
                                sentiment_df = pd.DataFrame.from_dict(topic_sentiments, orient='index')
                                sentiment_df.index.name = 'Topic'
                                sentiment_df = sentiment_df.reset_index()
                                
                                st.dataframe(sentiment_df, use_container_width=True)
                                
                                # Visualization
                                fig, ax = plt.subplots(figsize=(10, 5))
                                topics_list = sentiment_df['Topic'].astype(str)
                                colors = ['green' if cat == 'Positive' else 'red' if cat == 'Negative' else 'gray' 
                                         for cat in sentiment_df['category']]
                                ax.bar(topics_list, sentiment_df['avg_polarity'], color=colors, alpha=0.7)
                                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                                ax.set_xlabel('Topic')
                                ax.set_ylabel('Average Polarity')
                                ax.set_title('Sentiment Polarity by Topic (LDA)')
                                plt.xticks(rotation=0)
                                st.pyplot(fig)
                                
                            except Exception as e:
                                st.error(f"Error in LDA analysis: {str(e)}")
                        
                        # Perform NMF
                        if algorithm in ["NMF", "Both"]:
                            st.subheader("NMF (Non-negative Matrix Factorization)")
                            try:
                                nmf_model, nmf_output, nmf_topics, nmf_vectorizer = perform_nmf(
                                    cleaned_sentences, n_topics, n_top_words
                                )
                                
                                # Display topics
                                for topic in nmf_topics:
                                    with st.expander(f"Topic {topic['topic_num']}"):
                                        st.write("**Top Words:**")
                                        words_df = pd.DataFrame({
                                            'Word': topic['words'],
                                            'Weight': topic['weights']
                                        })
                                        st.dataframe(words_df, use_container_width=True)
                                
                                # Analyze sentiment by topic
                                st.subheader("Sentiment Analysis by Topic (NMF)")
                                topic_sentiments = analyze_topic_sentiment(cleaned_sentences, nmf_output)
                                
                                sentiment_df = pd.DataFrame.from_dict(topic_sentiments, orient='index')
                                sentiment_df.index.name = 'Topic'
                                sentiment_df = sentiment_df.reset_index()
                                
                                st.dataframe(sentiment_df, use_container_width=True)
                                
                                # Visualization
                                fig, ax = plt.subplots(figsize=(10, 5))
                                topics_list = sentiment_df['Topic'].astype(str)
                                colors = ['green' if cat == 'Positive' else 'red' if cat == 'Negative' else 'gray' 
                                         for cat in sentiment_df['category']]
                                ax.bar(topics_list, sentiment_df['avg_polarity'], color=colors, alpha=0.7)
                                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                                ax.set_xlabel('Topic')
                                ax.set_ylabel('Average Polarity')
                                ax.set_title('Sentiment Polarity by Topic (NMF)')
                                plt.xticks(rotation=0)
                                st.pyplot(fig)
                                
                            except Exception as e:
                                st.error(f"Error in NMF analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Narrative Nexus** - Advanced Text Analysis Tool | Built with Streamlit")
