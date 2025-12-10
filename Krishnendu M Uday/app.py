import streamlit as st
import os
from narrativenexus_utils import (
    ensure_data_dir,
    save_uploaded_file,
    get_sample_files,
    parse_preview,
    read_full_text,
    get_text_summary,
)
from narrativenexus_preprocess import clean_text, tokenize, SPACY_AVAILABLE
from narrativenexus_topic_modeling import TopicModelManager, GENSIM_AVAILABLE, SKLEARN_AVAILABLE

# Set folder to save uploaded samples
DATA_DIR = "sample_data"
ensure_data_dir(DATA_DIR)

st.set_page_config(page_title="NarrativeNexus", layout="wide", initial_sidebar_state="collapsed")

# Modern gradient styling with beautiful UI
st.markdown("""
    <style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    /* Main content card */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        max-width: 1200px;
        margin: 2rem auto;
    }
    
    /* Title styling */
    h1 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -1px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: white;
        border-radius: 10px;
        padding: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.4) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3) !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stAlert, div[data-baseweb="notification"] {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.12) 0%, rgba(56, 161, 105, 0.08) 100%) !important;
        border-radius: 10px !important;
        border: 1.5px solid rgba(56, 161, 105, 0.3) !important;
        box-shadow: none !important;
        padding: 0.85rem 1.1rem !important;
        max-width: fit-content !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(66, 153, 225, 0.12) 0%, rgba(49, 130, 206, 0.08) 100%) !important;
        border-radius: 10px !important;
        border: 1.5px solid rgba(49, 130, 206, 0.3) !important;
        box-shadow: none !important;
        padding: 0.85rem 1.1rem !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSuccess p, .stInfo p, .stAlert p, div[data-baseweb="notification"] p {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
        margin: 0 !important;
        line-height: 1.5 !important;
    }
    
    .stSuccess strong, .stInfo strong, .stAlert strong {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    .stSuccess div, .stInfo div, .stAlert div {
        color: #1a202c !important;
    }
    
    /* Target Streamlit's success and info icons */
    .stSuccess svg, .stInfo svg {
        width: 18px !important;
        height: 18px !important;
        color: #2d3748 !important;
    }
    
    /* File list items */
    .uploadedFileName {
        background: white;
        padding: 12px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 3px solid #667eea;
        display: flex;
        align-items: center;
    }
    
    /* Subheaders */
    h2, h3, h4 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3) !important;
        font-weight: 600 !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
    }
    
    h4 {
        font-size: 1.25rem !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Text area styling */
    textarea {
        color: #000000 !important;
        font-weight: 500 !important;
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.3s ease !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        line-height: 1.6 !important;
    }
    
    textarea:disabled {
        color: #000000 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #000000 !important;
        background: #fafafa !important;
    }
    
    textarea:hover {
        border-color: #cbd5e0 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
    }
    
    textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("NarrativeNexus")

tabs = st.tabs(["📤 Upload Files", "📊 File Analysis", "🔬 Text Processing", "🎯 Topic Modeling"])

# Upload Files Tab
with tabs[0]:
    st.markdown("### 📤 Upload Text Files")
    st.markdown("Drag & drop your files or browse • TXT, CSV, DOCX")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "csv", "docx"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file, DATA_DIR)
        st.success(f"✓ File '{uploaded_file.name}' uploaded successfully!")

    st.markdown("### 📁 Uploaded Files")
    sample_files = get_sample_files(DATA_DIR)
    if sample_files:
        for file in sample_files:
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(f"""
                    <div class='uploadedFileName'>
                        📄 {file}
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{file}"):
                    try:
                        os.remove(os.path.join(DATA_DIR, file))
                        st.rerun()
                    except:
                        pass
    else:
        st.info("📭 No files uploaded yet. Upload your first file above!")

# File Analysis Tab
with tabs[1]:
    st.markdown("### 📊 File Analysis")
    sample_files = get_sample_files(DATA_DIR)
    if sample_files:
        selected = st.selectbox("📁 Choose a file to analyze", sample_files, label_visibility="visible")
        if selected:
            file_path = os.path.join(DATA_DIR, selected)
            
            # Buttons in columns
            col1, col2 = st.columns(2)
            with col1:
                word_count_button = st.button("📈 Word Count Statistics", use_container_width=True)
            with col2:
                preview_button = st.button("👁️ Preview File", use_container_width=True)
            
            if word_count_button:
                with st.spinner("Analyzing file..."):
                    raw = read_full_text(file_path)
                    
                    if raw.startswith("[Error") or raw.startswith("[Cannot"):
                        st.error(raw)
                    else:
                        raw_words = raw.split()
                        raw_word_count = len(raw_words)
                        unique_words = len(set([w.lower() for w in raw_words]))
                        char_count = len(raw)
                        char_count_no_spaces = len(raw.replace(" ", ""))
                        sentence_count = raw.count('.') + raw.count('!') + raw.count('?')
                        
                        st.markdown("#### 📊 Statistics Overview")
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("📝 Total Words", f"{raw_word_count:,}")
                        with metric_col2:
                            st.metric("🔤 Unique Words", f"{unique_words:,}")
                        with metric_col3:
                            st.metric("📄 Characters", f"{char_count:,}")
                        
                        metric_col4, metric_col5, metric_col6 = st.columns(3)
                        with metric_col4:
                            st.metric("✏️ Chars (no spaces)", f"{char_count_no_spaces:,}")
                        with metric_col5:
                            st.metric("📖 Sentences", f"{sentence_count:,}")
                        with metric_col6:
                            avg_word_length = char_count_no_spaces / raw_word_count if raw_word_count > 0 else 0
                            st.metric("📏 Avg Word Length", f"{avg_word_length:.1f}")
            
            if preview_button:
                preview = parse_preview(file_path, max_lines=50)
                st.markdown("#### 👁️ File Preview")
                st.code(preview, language=None)
    else:
        st.info("📭 No files available. Upload files in the Upload Files tab!")

# Text Processing Tab
with tabs[2]:
    st.markdown("### 🔬 Advanced Text Processing")
    st.markdown("Preprocess text with tokenization, stopword removal, and lemmatization")
    
    sample_files = get_sample_files(DATA_DIR)
    if sample_files:
        selected = st.selectbox("📁 Choose a file to preprocess", sample_files, key="preprocess_select")
        if selected:
            file_path = os.path.join(DATA_DIR, selected)
            
            if st.button("🚀 Preprocess & Analyze", use_container_width=True):
                with st.spinner("🔄 Processing..."):
                    raw = read_full_text(file_path)
                    
                    if raw.startswith("[Error") or raw.startswith("[Cannot"):
                        st.error(raw)
                    else:
                        st.success(f"✓ File loaded: {len(raw)} characters")
                        
                        # Generate content summary
                        st.markdown("#### 📋 Content Summary")
                        summary = get_text_summary(raw)
                        st.info(f"**Analysis:** {summary}")
                        
                        cleaned = clean_text(raw)
                        toks = tokenize(cleaned)

                        # Show before/after comparison
                        st.markdown("#### 🔄 Text Transformation")
                        
                        col_before, col_after = st.columns(2)
                        
                        with col_before:
                            st.markdown("**📝 Original Text (first 500 chars)**")
                            original_preview = raw[:500].strip()
                            st.text_area(
                                "Original",
                                original_preview + ("..." if len(raw) > 500 else ""),
                                height=200,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        
                        with col_after:
                            st.markdown("**✨ Cleaned Text (first 500 chars)**")
                            cleaned_preview = cleaned[:500].strip()
                            st.text_area(
                                "Cleaned",
                                cleaned_preview + ("..." if len(cleaned) > 500 else ""),
                                height=200,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        
                        # Show full cleaned text in an expandable section
                        with st.expander("📄 View Full Cleaned Text", expanded=False):
                            st.markdown("**Complete cleaned output:**")
                            st.text_area(
                                "Full Cleaned Text",
                                cleaned,
                                height=400,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        
                        # Show cleaning statistics
                        st.markdown("#### 📈 Cleaning Statistics")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        with stat_col1:
                            reduction = ((len(raw) - len(cleaned)) / len(raw) * 100) if len(raw) > 0 else 0
                            st.metric("🔻 Size Reduction", f"{reduction:.1f}%")
                        with stat_col2:
                            st.metric("📏 Original Length", f"{len(raw):,} chars")
                        with stat_col3:
                            st.metric("✅ Cleaned Length", f"{len(cleaned):,} chars")
                        with stat_col4:
                            removed = len(raw.split()) - len(cleaned.split())
                            st.metric("🗑️ Words Removed", f"{removed:,}")

                        st.markdown("#### 📊 Token Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🔢 Total Tokens", f"{len(toks):,}")
                        with col2:
                            st.metric("🔣 Unique Tokens", f"{len(set(toks)):,}")

                        from collections import Counter
                        freq = Counter([t.lower() for t in toks])
                        top = freq.most_common(10)
                        
                        st.markdown("#### 🏆 Top 10 Frequent Tokens")
                        if top:
                            # Create a modern card-based representation
                            for i, (token, count) in enumerate(top, 1):
                                st.markdown(f"""
                                    <div style="
                                        background: rgba(255, 255, 255, 0.12);
                                        border-radius: 8px;
                                        padding: 0.6rem 1rem;
                                        margin: 0.4rem 0;
                                        display: flex;
                                        align-items: center;
                                        backdrop-filter: blur(10px);
                                        border: 1px solid rgba(255, 255, 255, 0.15);
                                    ">
                                        <div style="
                                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                            color: white;
                                            width: 28px;
                                            height: 28px;
                                            border-radius: 50%;
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                            font-weight: 700;
                                            font-size: 0.85rem;
                                            margin-right: 0.9rem;
                                            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
                                        ">{i}</div>
                                        <div style="flex: 1;">
                                            <span style="
                                                color: #ffffff;
                                                font-size: 1rem;
                                                font-weight: 600;
                                                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                                            ">{token}</span>
                                        </div>
                                        <div style="
                                            background: rgba(255, 255, 255, 0.18);
                                            padding: 0.25rem 0.7rem;
                                            border-radius: 15px;
                                            color: #ffffff;
                                            font-weight: 600;
                                            font-size: 0.85rem;
                                            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                                        ">{count}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("No tokens found.")

                        if SPACY_AVAILABLE:
                            st.info("✓ Using spaCy for advanced tokenization & lemmatization")
                        else:
                            st.warning("⚠ Using fallback preprocessing (spaCy not available)")
    else:
        st.info("📭 No files available. Upload files in the Upload Files tab!")

# Topic Modeling Tab
with tabs[3]:
    st.markdown("### 🎯 Topic Modeling")
    st.markdown("Discover hidden topics in your documents using LDA or NMF algorithms")
    
    # File requirements notice
    st.info("📋 **File Requirements:** For best results, use files with:\n- At least 50-100 documents/paragraphs\n- Substantial text content (500+ words)\n- Multiple distinct themes or topics")
    
    sample_files = get_sample_files(DATA_DIR)
    if sample_files:
        selected = st.selectbox("📁 Choose a file for topic modeling", sample_files, key="topic_select")
        
        if selected:
            file_path = os.path.join(DATA_DIR, selected)
            
            # Algorithm selection
            col1, col2 = st.columns(2)
            with col1:
                algorithm = st.radio(
                    "Choose Algorithm",
                    ["LDA (Latent Dirichlet Allocation)", "NMF (Non-negative Matrix Factorization)"],
                    help="LDA: Probabilistic, better for interpretable topics\nNMF: Faster, produces sparser topics"
                )
            
            with col2:
                num_topics = st.slider("Number of Topics", min_value=2, max_value=20, value=5, step=1)
                num_words = st.slider("Words per Topic", min_value=5, max_value=20, value=10, step=1)
            
            # Advanced LDA parameters (shown only for LDA)
            if "LDA" in algorithm:
                with st.expander("⚙️ Advanced LDA Parameters"):
                    col1, col2 = st.columns(2)
                    with col1:
                        passes = st.slider("Passes (Training Iterations)", min_value=1, max_value=20, value=4, step=1,
                                          help="Number of passes through the corpus during training")
                        alpha = st.selectbox("Alpha (Document-Topic Density)", 
                                           ["auto", "symmetric", "asymmetric"],
                                           help="Controls document-topic distribution. 'auto' learns from data.")
                    with col2:
                        iterations = st.slider("Iterations per Pass", min_value=50, max_value=500, value=100, step=50,
                                             help="Number of iterations per training pass")
                        beta = st.selectbox("Beta/Eta (Topic-Word Density)",
                                          ["auto", "symmetric"],
                                          help="Controls topic-word distribution. 'auto' learns from data.")
            else:
                passes = 4
                iterations = 100
                alpha = "auto"
                beta = "auto"
            
            use_lda = "LDA" in algorithm
            
            # Add model comparison and topic count comparison features
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                compare_models = st.checkbox("⚖️ Compare LDA vs NMF", 
                                            help="Train both models and compare their results side-by-side")
            
            with col2:
                use_ensemble = st.checkbox("🔀 Use Ensemble Approach",
                                          help="Combine LDA and NMF for better topic extraction")
            
            with col3:
                compare_topics = st.checkbox("🔬 Compare Topic Counts", 
                                            help="Compare different numbers of topics")
            
            if compare_topics:
                topic_counts = st.multiselect(
                    "Select topic counts to compare",
                    options=[3, 5, 7, 10, 15, 20],
                    default=[5, 7, 10],
                    help="Select 2-5 different topic counts to compare"
                )
            else:
                topic_counts = []
            
            # Check if required libraries are available
            if use_lda and not GENSIM_AVAILABLE:
                st.error("❌ Gensim is not installed. Install with: pip install gensim")
            elif not use_lda and not SKLEARN_AVAILABLE:
                st.error("❌ scikit-learn is not installed. Install with: pip install scikit-learn")
            else:
                # Topic comparison mode
                if compare_topics and len(topic_counts) > 1:
                    if st.button("🔬 Compare Topic Counts", use_container_width=True):
                        with st.spinner("Comparing different topic counts..."):
                            try:
                                # Read and preprocess text
                                raw = read_full_text(file_path)
                                
                                if raw.startswith("[Error") or raw.startswith("[Cannot"):
                                    st.error(raw)
                                else:
                                    # Prepare data
                                    cleaned = clean_text(raw)
                                    max_chars = 500000
                                    if len(cleaned) > max_chars:
                                        st.warning(f"Large text ({len(cleaned):,} chars). Using first {max_chars:,} characters.")
                                        cleaned = cleaned[:max_chars]
                                    
                                    para_splits = [p.strip() for p in cleaned.split('\n\n') if len(p.strip()) > 30]
                                    if len(para_splits) < 10:
                                        para_splits = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 20]
                                    
                                    # Limit documents
                                    if len(para_splits) > 500:
                                        import random
                                        random.seed(42)
                                        para_splits = random.sample(para_splits, 500)
                                    
                                    # Tokenize
                                    tokenized_docs = []
                                    for doc in para_splits:
                                        doc_tokens = tokenize(doc)
                                        if len(doc_tokens) > 3:
                                            tokenized_docs.append(doc_tokens)
                                    
                                    # Validate sufficient documents
                                    min_topics = min(topic_counts) if topic_counts else num_topics
                                    if len(tokenized_docs) < min_topics:
                                        st.error(f"❌ Not enough valid documents ({len(tokenized_docs)}) for {min_topics} topics. Try using a larger file.")
                                    else:
                                        # Compare different topic counts
                                        results = []
                                        for n_topics in sorted(topic_counts):
                                            if len(tokenized_docs) < n_topics:
                                                st.warning(f"⚠️ Skipping {n_topics} topics - not enough documents")
                                                continue
                                                
                                            st.info(f"Training model with {n_topics} topics...")
                                            
                                            tm = TopicModelManager()
                                            tm.create_dictionary(tokenized_docs, no_below=1, no_above=0.9)
                                            
                                            if len(tm.dictionary) == 0:
                                                st.error(f"❌ No valid terms for {n_topics} topics")
                                                continue
                                            
                                            tm.create_corpus(tokenized_docs)
                                            
                                            tm.train_lda_model(
                                                num_topics=n_topics,
                                                passes=passes,
                                                iterations=iterations,
                                                alpha=alpha if alpha == "auto" else alpha,
                                                eta=beta if beta == "auto" else beta,
                                                random_state=42
                                            )
                                            
                                            coherence = tm.compute_coherence_score(tokenized_docs)
                                            perplexity = tm.compute_perplexity()
                                            
                                            results.append({
                                                'topics': n_topics,
                                                'coherence': coherence,
                                                'perplexity': perplexity
                                            })
                                        
                                        # Display comparison results
                                        if len(results) == 0:
                                            st.error("❌ No models could be trained. File may be too small.")
                                        else:
                                            st.success("✅ Comparison complete!")
                                            st.markdown("#### 📊 Topic Count Comparison Results")
                                            
                                            # Create comparison visualization
                                            try:
                                                import matplotlib.pyplot as plt
                                                import numpy as np
                                                
                                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                                                
                                                topic_nums = [r['topics'] for r in results]
                                                coherences = [r['coherence'] for r in results]
                                                perplexities = [r['perplexity'] for r in results]
                                                
                                                # Coherence plot
                                                ax1.plot(topic_nums, coherences, marker='o', linewidth=2, markersize=8, color='#667eea')
                                                ax1.set_xlabel('Number of Topics', fontsize=12)
                                                ax1.set_ylabel('Coherence Score', fontsize=12)
                                                ax1.set_title('Coherence Score vs. Number of Topics', fontsize=14)
                                                ax1.grid(True, alpha=0.3)
                                                best_coherence_idx = coherences.index(max(coherences))
                                                ax1.axvline(x=topic_nums[best_coherence_idx], color='green', linestyle='--', alpha=0.5, label='Best')
                                                ax1.legend()
                                                
                                                # Perplexity plot
                                                ax2.plot(topic_nums, perplexities, marker='s', linewidth=2, markersize=8, color='#f093fb')
                                                ax2.set_xlabel('Number of Topics', fontsize=12)
                                                ax2.set_ylabel('Perplexity Score', fontsize=12)
                                                ax2.set_title('Perplexity vs. Number of Topics', fontsize=14)
                                                ax2.grid(True, alpha=0.3)
                                                best_perplexity_idx = perplexities.index(min(perplexities))
                                                ax2.axvline(x=topic_nums[best_perplexity_idx], color='green', linestyle='--', alpha=0.5, label='Best')
                                                ax2.legend()
                                                
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                plt.close()
                                            except:
                                                pass
                                            
                                            # Display results table
                                            st.markdown("#### 📋 Detailed Results")
                                            for r in results:
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric(f"🎯 {r['topics']} Topics", "")
                                                with col2:
                                                    st.metric("Coherence", f"{r['coherence']:.4f}")
                                                with col3:
                                                    st.metric("Perplexity", f"{r['perplexity']:.2f}")
                                            
                                            # Recommendation with detailed explanation
                                            best_coherence = max(results, key=lambda x: x['coherence'])
                                            best_perplexity = min(results, key=lambda x: x['perplexity'])
                                            
                                            st.success(f"💡 **Recommendation:** {best_coherence['topics']} topics shows the best coherence score ({best_coherence['coherence']:.4f})")
                                            
                                            # Detailed explanation section
                                            st.markdown("---")
                                            st.markdown("#### 🧠 Understanding the Metrics & Recommendation")
                                            
                                            # Explain Coherence
                                            st.markdown("##### 📊 **What is Coherence Score?**")
                                            st.info("""
                                            **Coherence Score** measures how semantically similar the top words in each topic are to each other. 
                                            
                                            - **Range:** Typically between -1 to 1 (higher is better)
                                            - **Good Score:** Above 0.4 indicates well-formed, interpretable topics
                                            - **Moderate Score:** 0.3-0.4 suggests reasonable topic quality
                                            - **Low Score:** Below 0.3 may indicate unclear or overlapping topics
                                            
                                            **Why it matters:** High coherence means the words in each topic naturally "go together" and form meaningful themes that humans can understand.
                                            """)
                                            
                                            # Explain Perplexity
                                            st.markdown("##### 📉 **What is Perplexity?**")
                                            st.info("""
                                            **Perplexity** measures how well the model predicts a sample of data. It's based on the likelihood of the model generating the observed documents.
                                            
                                            - **Range:** Positive numbers (lower is better)
                                            - **Interpretation:** Lower perplexity means the model is less "surprised" by the data
                                            - **Note:** Perplexity alone doesn't guarantee interpretable topics
                                            
                                            **Why it matters:** Low perplexity indicates the model has learned meaningful patterns, but it should be balanced with coherence for best results.
                                            """)
                                            
                                            # Recommendation reasoning
                                            st.markdown("##### 💡 **Why This Recommendation?**")
                                            
                                            coherence_val = best_coherence['coherence']
                                            perplexity_val = best_coherence['perplexity']
                                            
                                            # Build explanation
                                            explanation_parts = []
                                            
                                            # Coherence reasoning
                                            if coherence_val > 0.4:
                                                coherence_quality = "**excellent**"
                                                coherence_explanation = "The topics are highly interpretable with semantically related words."
                                            elif coherence_val > 0.3:
                                                coherence_quality = "**good**"
                                                coherence_explanation = "The topics show reasonable semantic coherence and are generally interpretable."
                                            elif coherence_val > 0.2:
                                                coherence_quality = "**moderate**"
                                                coherence_explanation = "The topics have some coherence but may overlap or be less distinct."
                                            else:
                                                coherence_quality = "**low**"
                                                coherence_explanation = "The topics may be unclear or need refinement."
                                            
                                            st.markdown(f"""
                                            **🎯 Selected: {best_coherence['topics']} Topics**
                                            
                                            **Coherence Score: {coherence_val:.4f}** ({coherence_quality})
                                            - {coherence_explanation}
                                            
                                            **Perplexity: {perplexity_val:.2f}**
                                            - {"Lower perplexity indicates better model fit to the data." if perplexity_val == best_perplexity['perplexity'] else "Perplexity is within acceptable range for this topic count."}
                                            """)
                                            
                                            # Compare with other configurations
                                            if len(results) > 1:
                                                st.markdown("**Why not other configurations?**")
                                                for r in results:
                                                    if r['topics'] != best_coherence['topics']:
                                                        coherence_diff = best_coherence['coherence'] - r['coherence']
                                                        if coherence_diff > 0.01:
                                                            st.markdown(f"- **{r['topics']} topics** (Coherence: {r['coherence']:.4f}): {abs(coherence_diff):.4f} points lower coherence means less interpretable topics")
                                                        elif coherence_diff < -0.01:
                                                            st.markdown(f"- **{r['topics']} topics** (Coherence: {r['coherence']:.4f}): Slightly higher coherence, but {best_coherence['topics']} topics provide better balance")
                                                        else:
                                                            st.markdown(f"- **{r['topics']} topics** (Coherence: {r['coherence']:.4f}): Similar performance, {best_coherence['topics']} topics chosen for optimal granularity")
                                            
                                            # Practical advice
                                            st.markdown("##### 🎓 **Practical Guidance**")
                                            if coherence_val > 0.35:
                                                st.success(f"✅ **Excellent choice!** With a coherence of {coherence_val:.4f}, your {best_coherence['topics']} topics will produce meaningful, interpretable themes from your text.")
                                            elif coherence_val > 0.25:
                                                st.info(f"👍 **Good choice!** {best_coherence['topics']} topics should work well. If topics seem unclear, consider adjusting preprocessing or trying fewer topics.")
                                            else:
                                                st.warning(f"⚠️ **Moderate performance.** You may want to try different preprocessing settings, adjust the number of topics, or use a larger/different text corpus for better results.")
                                    
                            except Exception as e:
                                st.error(f"❌ Error during comparison: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                
                # LDA vs NMF comparison mode
                elif compare_models:
                    if st.button("⚖️ Compare LDA vs NMF", use_container_width=True):
                        with st.spinner("Comparing LDA and NMF models..."):
                            try:
                                # Read and preprocess text
                                raw = read_full_text(file_path)
                                
                                if raw.startswith("[Error") or raw.startswith("[Cannot"):
                                    st.error(raw)
                                else:
                                    # Prepare data
                                    cleaned = clean_text(raw)
                                    max_chars = 500000
                                    if len(cleaned) > max_chars:
                                        st.warning(f"Large text ({len(cleaned):,} chars). Using first {max_chars:,} characters.")
                                        cleaned = cleaned[:max_chars]
                                    
                                    para_splits = [p.strip() for p in cleaned.split('\n\n') if len(p.strip()) > 30]
                                    if len(para_splits) < 10:
                                        para_splits = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 20]
                                    
                                    # Limit documents for comparison
                                    if len(para_splits) > 500:
                                        import random
                                        random.seed(42)
                                        para_splits = random.sample(para_splits, 500)
                                    
                                    st.info(f"📊 Comparing models on {len(para_splits)} documents with {num_topics} topics")
                                    
                                    # Tokenize first to validate we have enough content
                                    tokenized_docs = []
                                    for doc in para_splits:
                                        doc_tokens = tokenize(doc)
                                        if len(doc_tokens) > 3:
                                            tokenized_docs.append(doc_tokens)
                                    
                                    # Validate we have enough tokenized documents
                                    if len(tokenized_docs) < num_topics:
                                        st.error(f"❌ Not enough valid documents ({len(tokenized_docs)}) for {num_topics} topics. Try reducing topic count or using a larger file.")
                                        
                                        # Still show algorithm comparison info
                                        st.markdown("---")
                                        st.markdown("### ℹ️ Algorithm Comparison")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("#### 🔵 LDA (Recommended)")
                                            st.markdown("""
                                            - ✅ Probabilistic model
                                            - ✅ Interpretable topics
                                            - ✅ Handles overlapping themes
                                            - ✅ Better for smaller datasets
                                            - ⚠️ Slower training
                                            """)
                                        
                                        with col2:
                                            st.markdown("#### 🟠 NMF (Alternative)")
                                            st.markdown("""
                                            - ✅ Fast computation
                                            - ✅ Sparse, distinct topics
                                            - ✅ Good for large datasets
                                            - ⚠️ Less interpretable
                                            - ⚠️ Topics may be too separated
                                            """)
                                    else:
                                        # Prepare for both models
                                        col1, col2 = st.columns(2)
                                        
                                        # Train LDA
                                        with col1:
                                            st.markdown("### 🔵 LDA Model")
                                            with st.spinner("Training LDA..."):
                                                tm_lda = TopicModelManager()
                                                
                                                tm_lda.create_dictionary(tokenized_docs, no_below=1, no_above=0.9)
                                                
                                                # Validate dictionary
                                                if len(tm_lda.dictionary) == 0:
                                                    st.error("❌ No valid terms in dictionary. Text may be too short or heavily filtered.")
                                                else:
                                                    tm_lda.create_corpus(tokenized_docs)
                                                    
                                                    tm_lda.train_lda_model(
                                                        num_topics=num_topics,
                                                        passes=passes,
                                                        iterations=iterations,
                                                        alpha=alpha if alpha == "auto" else alpha,
                                                        eta=beta if beta == "auto" else beta,
                                                        random_state=42
                                                    )
                                                    
                                                    lda_coherence = tm_lda.compute_coherence_score(tokenized_docs)
                                                    lda_perplexity = tm_lda.compute_perplexity()
                                                    lda_topics = tm_lda.get_lda_topics(num_words=num_words)
                                                    
                                                    st.markdown("**Top Topics:**")
                                                    for topic_id, words in lda_topics[:3]:
                                                        word_list = ", ".join([f"{word}" for word, _ in words[:5]])
                                                        st.markdown(f"**Topic {topic_id + 1}:** {word_list}")
                                                    
                                                    st.markdown("---")
                                                    st.markdown("**📊 LDA Metrics:**")
                                                    st.metric("📈 Coherence", f"{lda_coherence:.4f}")
                                                    st.metric("📉 Perplexity", f"{lda_perplexity:.2f}")
                                                    st.metric("📚 Vocab", f"{len(tm_lda.dictionary)}")
                                        
                                        # Train NMF
                                        with col2:
                                            st.markdown("### 🟠 NMF Model")
                                            with st.spinner("Training NMF..."):
                                                tm_nmf = TopicModelManager()
                                                
                                                tm_nmf.train_nmf_model(
                                                    documents=para_splits,
                                                    num_topics=num_topics,
                                                    max_features=min(1000, len(para_splits) * 5),
                                                    random_state=42
                                                )
                                                
                                                nmf_topics = tm_nmf.get_nmf_topics(num_words=num_words)
                                                
                                                # Compute coherence for NMF
                                                nmf_coherence = tm_nmf.compute_nmf_coherence(tokenized_docs)
                                                
                                                st.markdown("**Top Topics:**")
                                                for topic_id, words in nmf_topics[:3]:
                                                    word_list = ", ".join([f"{word}" for word, _ in words[:5]])
                                                    st.markdown(f"**Topic {topic_id + 1}:** {word_list}")
                                                
                                                st.markdown("---")
                                                st.markdown("**📊 NMF Metrics:**")
                                                st.metric("📈 Coherence", f"{nmf_coherence:.4f}")
                                                st.metric("🎯 Topics Extracted", f"{num_topics}")
                                                st.metric("📚 Features Used", f"{len(tm_nmf.vectorizer.get_feature_names_out())}")
                                        
                                        # Detailed Comparison (only if LDA succeeded)
                                        if len(tm_lda.dictionary) > 0:
                                            st.markdown("---")
                                            st.markdown("### 📊 Detailed Comparison")
                                            
                                            # Create comparison visualization
                                            try:
                                                import matplotlib.pyplot as plt
                                                import numpy as np
                                                
                                                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                                                
                                                # Coherence comparison
                                                ax1 = axes[0]
                                                models = ['LDA', 'NMF']
                                                coherences = [lda_coherence, nmf_coherence]
                                                colors = ['#667eea', '#f093fb']
                                                bars1 = ax1.bar(models, coherences, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
                                                ax1.set_ylabel('Coherence Score', fontsize=12, fontweight='bold')
                                                ax1.set_title('Coherence Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
                                                ax1.set_ylim(0, max(coherences) * 1.3)
                                                ax1.grid(axis='y', alpha=0.3, linestyle='--')
                                                
                                                # Add value labels on bars
                                                for bar, value in zip(bars1, coherences):
                                                    height = bar.get_height()
                                                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                                                            f'{value:.4f}',
                                                            ha='center', va='bottom', fontsize=11, fontweight='bold')
                                                
                                                # Mark the better one
                                                better_coherence_idx = 0 if lda_coherence > nmf_coherence else 1
                                                bars1[better_coherence_idx].set_edgecolor('gold')
                                                bars1[better_coherence_idx].set_linewidth(3)
                                                
                                                # Perplexity comparison (LDA only)
                                                ax2 = axes[1]
                                                ax2.bar(['LDA'], [abs(lda_perplexity)], color='#667eea', alpha=0.8, edgecolor='white', linewidth=2)
                                                ax2.set_ylabel('Perplexity (Absolute)', fontsize=12, fontweight='bold')
                                                ax2.set_title('Perplexity\n(Lower is Better - LDA Only)', fontsize=14, fontweight='bold')
                                                ax2.grid(axis='y', alpha=0.3, linestyle='--')
                                                ax2.text(0, abs(lda_perplexity), f'{lda_perplexity:.2f}',
                                                        ha='center', va='bottom', fontsize=11, fontweight='bold')
                                                ax2.set_xlim(-0.5, 0.5)
                                                
                                                # Add note
                                                ax2.text(0, -abs(lda_perplexity) * 0.15, 
                                                        'NMF does not use\nprobabilistic modeling',
                                                        ha='center', fontsize=9, style='italic', color='gray')
                                                
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                plt.close()
                                            except Exception as e:
                                                st.warning(f"Could not create visualization: {str(e)}")
                                            
                                            # Metrics comparison table
                                            st.markdown("#### 📋 Metrics Breakdown")
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.markdown("**Metric**")
                                                st.markdown("Coherence Score")
                                                st.markdown("Perplexity")
                                                st.markdown("Vocabulary/Features")
                                            with col2:
                                                st.markdown("**🔵 LDA**")
                                                st.markdown(f"{lda_coherence:.4f}")
                                                st.markdown(f"{lda_perplexity:.2f}")
                                                st.markdown(f"{len(tm_lda.dictionary)}")
                                            with col3:
                                                st.markdown("**🟠 NMF**")
                                                st.markdown(f"{nmf_coherence:.4f}")
                                                st.markdown("N/A")
                                                st.markdown(f"{len(tm_nmf.vectorizer.get_feature_names_out())}")
                                            
                                            st.markdown("---")
                                            
                                            # Detailed recommendation with explanation
                                            st.markdown("### 💡 Which Model Should You Choose?")
                                            
                                            coherence_diff = abs(lda_coherence - nmf_coherence)
                                            lda_better_coherence = lda_coherence > nmf_coherence
                                            
                                            if lda_better_coherence:
                                                winner = "LDA"
                                                winner_icon = "🔵"
                                                winner_coherence = lda_coherence
                                                loser = "NMF"
                                                loser_coherence = nmf_coherence
                                            else:
                                                winner = "NMF"
                                                winner_icon = "🟠"
                                                winner_coherence = nmf_coherence
                                                loser = "LDA"
                                                loser_coherence = lda_coherence
                                            
                                            # Show recommendation
                                            if coherence_diff > 0.05:
                                                st.success(f"✅ **Strong Recommendation: {winner_icon} {winner}**")
                                                st.markdown(f"""
                                                **Why {winner}?**
                                                - **Coherence Score:** {winner_coherence:.4f} vs {loser_coherence:.4f}
                                                - **Difference:** {coherence_diff:.4f} ({coherence_diff/loser_coherence*100:.1f}% better)
                                                - **Interpretation:** {winner} produces significantly more interpretable and semantically coherent topics
                                                """)
                                            elif coherence_diff > 0.02:
                                                st.info(f"👍 **Recommendation: {winner_icon} {winner} (Moderate Advantage)**")
                                                st.markdown(f"""
                                                **Why {winner}?**
                                                - **Coherence Score:** {winner_coherence:.4f} vs {loser_coherence:.4f}
                                                - **Difference:** {coherence_diff:.4f} (slightly better)
                                                - **Interpretation:** {winner} has a modest edge in topic coherence
                                                """)
                                            else:
                                                st.info("⚖️ **Both Models Perform Similarly**")
                                                st.markdown(f"""
                                                **Coherence Scores:**
                                                - LDA: {lda_coherence:.4f}
                                                - NMF: {nmf_coherence:.4f}
                                                - Difference: {coherence_diff:.4f} (negligible)
                                                
                                                **Choose based on your needs:**
                                                - **LDA:** Better for probabilistic interpretation and document classification
                                                - **NMF:** Faster training and sparser topics
                                                """)
                                            
                                            # Additional context
                                            st.markdown("---")
                                            st.markdown("#### 🎓 Understanding the Results")
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.markdown("**📈 Coherence Score**")
                                                if winner_coherence > 0.4:
                                                    quality = "Excellent"
                                                    emoji = "🌟"
                                                elif winner_coherence > 0.3:
                                                    quality = "Good"
                                                    emoji = "👍"
                                                elif winner_coherence > 0.2:
                                                    quality = "Moderate"
                                                    emoji = "⚠️"
                                                else:
                                                    quality = "Low"
                                                    emoji = "📉"
                                                
                                                st.info(f"{emoji} **{quality}** ({winner_coherence:.4f})\n\nThe winning model's topics have {quality.lower()} semantic coherence, meaning the words in each topic naturally belong together.")
                                            
                                            with col2:
                                                st.markdown("**📉 Perplexity (LDA Only)**")
                                                if lda_perplexity < -7:
                                                    perp_quality = "Excellent"
                                                    perp_emoji = "🌟"
                                                elif lda_perplexity < -6:
                                                    perp_quality = "Good"
                                                    perp_emoji = "👍"
                                                elif lda_perplexity < -5:
                                                    perp_quality = "Moderate"
                                                    perp_emoji = "⚠️"
                                                else:
                                                    perp_quality = "Needs Improvement"
                                                    perp_emoji = "📉"
                                                
                                                st.info(f"{perp_emoji} **{perp_quality}** ({lda_perplexity:.2f})\n\nLower (more negative) is better. This indicates how well LDA predicts the document patterns.")
                                            
                                            # Practical advice
                                            st.markdown("#### 🎯 Practical Recommendations")
                                            if lda_better_coherence and lda_coherence > 0.35:
                                                st.success("✅ **Use LDA** - It has superior coherence and perplexity metrics, making it ideal for interpretable topic modeling.")
                                            elif not lda_better_coherence and nmf_coherence > 0.35:
                                                st.success("✅ **Use NMF** - It has better coherence and faster training, making it ideal for your dataset.")
                                            else:
                                                st.info("💡 **Consider:**\n- LDA for better probabilistic interpretation\n- NMF for faster processing and clearer topic separation\n- Both models perform reasonably well")
                                    
                            except Exception as e:
                                st.error(f"❌ Error during model comparison: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                
                # Ensemble mode
                elif use_ensemble:
                    if st.button("🔀 Extract Topics with Ensemble", use_container_width=True):
                        with st.spinner("Training ensemble model (LDA + NMF)..."):
                            try:
                                # Read and preprocess text
                                raw = read_full_text(file_path)
                                
                                if raw.startswith("[Error") or raw.startswith("[Cannot"):
                                    st.error(raw)
                                else:
                                    # Prepare data
                                    cleaned = clean_text(raw)
                                    max_chars = 500000
                                    if len(cleaned) > max_chars:
                                        st.warning(f"Large text ({len(cleaned):,} chars). Using first {max_chars:,} characters.")
                                        cleaned = cleaned[:max_chars]
                                    
                                    para_splits = [p.strip() for p in cleaned.split('\n\n') if len(p.strip()) > 30]
                                    if len(para_splits) < 10:
                                        para_splits = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 20]
                                    
                                    # Limit documents
                                    if len(para_splits) > 500:
                                        import random
                                        random.seed(42)
                                        para_splits = random.sample(para_splits, 500)
                                    
                                    # Tokenize for LDA
                                    tokenized_docs = []
                                    for doc in para_splits:
                                        doc_tokens = tokenize(doc)
                                        if len(doc_tokens) > 3:
                                            tokenized_docs.append(doc_tokens)
                                    
                                    # Validate sufficient documents
                                    if len(tokenized_docs) < num_topics:
                                        st.error(f"❌ Not enough valid documents ({len(tokenized_docs)}) for {num_topics} topics. Try reducing topic count or using a larger file.")
                                    else:
                                        st.info(f"🔀 Training ensemble model on {len(tokenized_docs)} documents...")
                                        
                                        # Initialize and train ensemble
                                        tm = TopicModelManager()
                                        ensemble_topics = tm.get_ensemble_topics(
                                            tokenized_docs=tokenized_docs,
                                            documents=para_splits,
                                            num_topics=num_topics,
                                            num_words=num_words
                                        )
                                        
                                        st.success("✅ Ensemble model trained successfully!")
                                        st.info("📊 Topics below combine insights from both LDA (probabilistic) and NMF (sparse) models")
                                        
                                        # Display ensemble topics
                                        st.markdown("#### 🔀 Ensemble Topics (LDA + NMF)")
                                        for topic_id, words in ensemble_topics:
                                            word_list = ", ".join([f"{word} ({weight:.3f})" for word, weight in words[:5]])
                                            
                                            st.markdown(f"""
                                                <div style="
                                                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(240, 147, 251, 0.15));
                                                    border-radius: 10px;
                                                    padding: 1rem;
                                                    margin: 0.5rem 0;
                                                    border-left: 4px solid #667eea;
                                                    border-right: 4px solid #f093fb;
                                                ">
                                                    <h4 style="color: #ffffff; margin: 0 0 0.5rem 0;">🔀 Topic {topic_id + 1}</h4>
                                                    <p style="color: #ffffff; margin: 0; font-size: 0.95rem;">{word_list}</p>
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                                            with st.expander(f"View all {num_words} words for Topic {topic_id + 1}"):
                                                for word, weight in words:
                                                    st.write(f"• {word}: {weight:.4f}")
                                        
                                        # Display statistics
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("📄 Documents", f"{len(tokenized_docs):,}")
                                        with col2:
                                            st.metric("🎯 Topics", f"{num_topics}")
                                        with col3:
                                            st.metric("🔀 Models Combined", "2")
                                        
                                        st.info("💡 **Ensemble Advantage:** Combines LDA's probabilistic interpretability with NMF's sparse clarity")
                                    
                            except Exception as e:
                                st.error(f"❌ Error during ensemble training: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                
                # Initialize extract_button to avoid NameError
                extract_button = False
                
                # Regular extraction mode - only show if no comparison modes are active
                if not compare_models and not use_ensemble and not compare_topics:
                    # Center-aligned button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        extract_button = st.button("🚀 Extract Topics", use_container_width=True)
                        if st.session_state.get('show_results', False):
                            if st.button("🔄 Clear Results", use_container_width=True):
                                st.session_state['show_results'] = False
                                st.rerun()
                
                # Use session state to preserve results across reruns
                if extract_button:
                    st.session_state['show_results'] = True
                
                if st.session_state.get('show_results', False):
                    with st.spinner(f"Training {algorithm.split()[0]} model..."):
                        try:
                            # Read and preprocess text
                            raw = read_full_text(file_path)
                            
                            if raw.startswith("[Error") or raw.startswith("[Cannot"):
                                st.error(raw)
                            else:
                                # Clean and tokenize
                                cleaned = clean_text(raw)
                                
                                # Initialize topic model manager
                                tm = TopicModelManager()
                                
                                if use_lda:
                                    # LDA Processing
                                    st.info("📚 Processing with LDA (Optimized)...")
                                    
                                    # Smart document splitting based on file size
                                    # For large files, use chunking to limit processing
                                    max_chars = 500000  # Process up to 500k characters
                                    if len(raw) > max_chars:
                                        st.warning(f"⚡ Large file detected ({len(raw):,} chars). Processing first {max_chars:,} characters for faster training.")
                                        raw_processed = raw[:max_chars]
                                    else:
                                        raw_processed = raw
                                    
                                    # Split into documents - use multiple delimiters for better chunking
                                    docs = []
                                    # Try paragraph splits first (double newlines)
                                    para_splits = [p.strip() for p in raw_processed.split('\n\n') if len(p.strip()) > 30]
                                    
                                    if len(para_splits) >= num_topics * 2:
                                        docs = para_splits
                                    else:
                                        # Fall back to sentence splits
                                        docs = [s.strip() for s in raw_processed.split('.') if len(s.strip()) > 20]
                                    
                                    # Limit documents to improve speed (max 500 docs)
                                    if len(docs) > 500:
                                        st.info(f"🚀 Sampling {500} documents from {len(docs)} for faster training")
                                        import random
                                        random.seed(42)
                                        docs = random.sample(docs, 500)
                                    
                                    # Tokenize each document with progress using status container
                                    tokenized_docs = []
                                    
                                    with st.status("📊 Processing with LDA (Optimized)...", expanded=True) as status:
                                        # Tokenization phase
                                        st.write(f"📄 Tokenizing {len(docs)} documents...")
                                        progress_bar = st.progress(0)
                                        
                                        for idx, doc in enumerate(docs):
                                            doc_cleaned = clean_text(doc)
                                            doc_tokens = tokenize(doc_cleaned)
                                            if len(doc_tokens) > 3:  # Only keep docs with at least 3 tokens
                                                tokenized_docs.append(doc_tokens)
                                            
                                            # Update progress every 10%
                                            if idx % max(1, len(docs) // 10) == 0:
                                                progress = idx / len(docs)
                                                progress_bar.progress(progress)
                                        
                                        progress_bar.progress(1.0)
                                        st.write(f"✅ Tokenized {len(tokenized_docs)} documents")
                                        
                                        if len(tokenized_docs) < num_topics:
                                            status.update(label="❌ Processing Failed", state="error", expanded=False)
                                            st.warning(f"⚠️ Not enough documents ({len(tokenized_docs)}) for {num_topics} topics. Try reducing the number of topics.")
                                        else:
                                            # Create dictionary and corpus
                                            st.write("📖 Building dictionary and corpus...")
                                            tm.create_dictionary(tokenized_docs, no_below=2, no_above=0.8)
                                            tm.create_corpus(tokenized_docs)
                                            
                                            # Save dictionary and corpus
                                            dict_path, corpus_path = tm.save_dictionary_and_corpus(prefix=f"lda_{num_topics}")
                                            
                                            # Train LDA model with optimized parameters
                                            st.write(f"🎯 Training LDA model (this may take 10-30 seconds)...")
                                            
                                            # Convert alpha and beta parameters
                                            alpha_val = alpha if alpha == "auto" else alpha
                                            beta_val = beta if beta == "auto" else beta
                                            
                                            tm.train_lda_model(
                                                num_topics=num_topics, 
                                                passes=passes, 
                                                iterations=iterations,
                                                alpha=alpha_val,
                                                eta=beta_val,
                                                random_state=42
                                            )
                                            
                                            # Save model
                                            model_path = tm.save_lda_model(filename=f"lda_{num_topics}_topics")
                                            
                                            status.update(label="✅ LDA Model Training Complete!", state="complete", expanded=False)
                                    
                                    if len(tokenized_docs) >= num_topics:
                                        st.success(f"✅ LDA model trained successfully in {passes} passes!")
                                        st.markdown(f"**Processed:** {len(tokenized_docs)} documents | **Vocab:** {len(tm.dictionary)} words")
                                        
                                        # Display topics
                                        st.markdown("#### 📊 Discovered Topics")
                                        topics = tm.get_lda_topics(num_words=num_words)
                                        
                                        # Add topic-word visualization option
                                        show_word_viz = st.checkbox("📊 Show Topic-Word Distribution Charts", value=False)
                                        
                                        if show_word_viz:
                                            try:
                                                fig = tm.create_topic_word_visualization(num_words=num_words, use_lda=True)
                                                st.pyplot(fig)
                                                import matplotlib.pyplot as plt
                                                plt.close(fig)
                                            except Exception as e:
                                                st.warning(f"Could not create visualization: {str(e)}")
                                        
                                        for topic_id, words in topics:
                                            # Create topic visualization
                                            word_list = ", ".join([f"{word} ({prob:.3f})" for word, prob in words[:5]])
                                            
                                            st.markdown(f"""
                                                <div style="
                                                    background: rgba(255, 255, 255, 0.12);
                                                    border-radius: 10px;
                                                    padding: 1rem;
                                                    margin: 0.5rem 0;
                                                    border-left: 4px solid #667eea;
                                                ">
                                                    <h4 style="color: #ffffff; margin: 0 0 0.5rem 0;">🏷️ Topic {topic_id + 1}</h4>
                                                    <p style="color: #ffffff; margin: 0; font-size: 0.95rem;">{word_list}</p>
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # Show all words in expander
                                            with st.expander(f"View all {num_words} words for Topic {topic_id + 1}"):
                                                for word, prob in words:
                                                    st.write(f"• {word}: {prob:.4f}")
                                        
                                        # Compute and display evaluation metrics
                                        st.markdown("#### 📊 Model Evaluation")
                                        col1, col2, col3, col4 = st.columns(4)
                                        
                                        with col1:
                                            try:
                                                coherence = tm.compute_coherence_score(tokenized_docs)
                                                st.metric("📈 Coherence Score", f"{coherence:.4f}", 
                                                         help="Higher = better topic quality (typically 0.3-0.7)")
                                            except:
                                                st.metric("📈 Coherence Score", "N/A")
                                        
                                        with col2:
                                            try:
                                                perplexity = tm.compute_perplexity()
                                                st.metric("📉 Perplexity", f"{perplexity:.2f}",
                                                         help="Lower = better model fit (more negative is better)")
                                            except:
                                                st.metric("📉 Perplexity", "N/A")
                                        
                                        with col3:
                                            st.metric("📚 Vocabulary", f"{len(tm.dictionary):,}")
                                        
                                        with col4:
                                            st.metric("📄 Documents", f"{len(tokenized_docs):,}")
                                        
                                        # Topic distribution visualization
                                        try:
                                            import matplotlib.pyplot as plt
                                            import numpy as np
                                            
                                            st.markdown("#### 📈 Topic Distribution Across Documents")
                                            
                                            topic_dist = tm.get_topic_distribution_matrix()
                                            topic_dist_array = np.array(topic_dist)
                                            
                                            # Create heatmap
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            im = ax.imshow(topic_dist_array.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
                                            
                                            ax.set_xlabel('Document Index', fontsize=12)
                                            ax.set_ylabel('Topic ID', fontsize=12)
                                            ax.set_title('Topic Distribution Heatmap', fontsize=14, pad=20)
                                            
                                            # Add colorbar
                                            cbar = plt.colorbar(im, ax=ax)
                                            cbar.set_label('Topic Probability', rotation=270, labelpad=20)
                                            
                                            # Set y-axis ticks
                                            ax.set_yticks(range(num_topics))
                                            ax.set_yticklabels([f'Topic {i+1}' for i in range(num_topics)])
                                            
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.close()
                                            
                                            # Show topic prevalence
                                            st.markdown("#### 📊 Topic Prevalence")
                                            topic_prevalence = topic_dist_array.mean(axis=0)
                                            
                                            fig2, ax2 = plt.subplots(figsize=(10, 4))
                                            bars = ax2.bar(range(num_topics), topic_prevalence, color='#667eea')
                                            ax2.set_xlabel('Topic ID', fontsize=12)
                                            ax2.set_ylabel('Average Probability', fontsize=12)
                                            ax2.set_title('Average Topic Prevalence Across All Documents', fontsize=14)
                                            ax2.set_xticks(range(num_topics))
                                            ax2.set_xticklabels([f'Topic {i+1}' for i in range(num_topics)])
                                            
                                            # Add value labels on bars
                                            for bar in bars:
                                                height = bar.get_height()
                                                ax2.text(bar.get_x() + bar.get_width()/2., height,
                                                        f'{height:.3f}',
                                                        ha='center', va='bottom', fontsize=9)
                                            
                                            plt.tight_layout()
                                            st.pyplot(fig2)
                                            plt.close()
                                            
                                        except Exception as e:
                                            st.warning(f"Could not generate visualization: {str(e)}")
                                        
                                        # Display statistics
                                        stats = tm.get_stats()
                                        st.markdown("#### 📊 Model Statistics")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("📚 Vocabulary Size", f"{stats.get('vocab_size', 0):,}")
                                        with col2:
                                            st.metric("📄 Documents", f"{len(tokenized_docs):,}")
                                        with col3:
                                            st.metric("🎯 Topics", f"{num_topics}")
                                        
                                        # Model export and download
                                        st.markdown("---")
                                        st.markdown("#### 💾 Save Model & Results")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            # Export configuration
                                            import json
                                            config = tm.export_model_config()
                                            config_json = json.dumps(config, indent=2)
                                            st.download_button(
                                                label="📥 Download Model Config",
                                                data=config_json,
                                                file_name=f"lda_model_config_{num_topics}topics.json",
                                                mime="application/json",
                                                help="Download model parameters and settings"
                                            )
                                        
                                        with col2:
                                            # Export topics as CSV
                                            topics_text = "Topic ID,Word,Probability\n"
                                            for topic_id, words in topics:
                                                for word, prob in words:
                                                    topics_text += f"{topic_id + 1},{word},{prob:.6f}\n"
                                            
                                            st.download_button(
                                                label="📥 Download Topics (CSV)",
                                                data=topics_text,
                                                file_name=f"lda_topics_{num_topics}topics.csv",
                                                mime="text/csv",
                                                help="Download extracted topics as CSV"
                                            )
                                        
                                        # Test on new samples
                                        st.markdown("---")
                                        st.markdown("#### 🧪 Test Model on New Text")
                                        
                                        test_text = st.text_area(
                                            "Enter new text to analyze with trained model",
                                            height=150,
                                            placeholder="Paste any text here to see its topic distribution..."
                                        )
                                        
                                        if st.button("🔍 Analyze New Text") and test_text.strip():
                                            # Create a status container for better loading visibility
                                            with st.status("🔍 Analyzing your text...", expanded=True) as status:
                                                try:
                                                    st.write("📝 Preprocessing text...")
                                                    st.write("🧠 Applying trained model...")
                                                    st.write("📊 Computing topic distribution...")
                                                    
                                                    topic_dist = tm.predict_topics_for_text(test_text, use_lda=True)
                                                    
                                                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                                                    
                                                    st.markdown("**Topic Distribution:**")
                                                    for topic_id, prob in sorted(topic_dist, key=lambda x: x[1], reverse=True):
                                                        if prob > 0.05:  # Only show topics with >5% probability
                                                            # Convert numpy float to Python float
                                                            prob_value = float(prob)
                                                            st.progress(prob_value, text=f"Topic {topic_id + 1}: {prob_value:.2%}")
                                                    
                                                    # Show top matching topic with detailed explanation
                                                    top_topic = max(topic_dist, key=lambda x: x[1])
                                                    top_topic_id = top_topic[0]
                                                    top_prob = float(top_topic[1])
                                                    st.success(f"🎯 Primary Topic: **Topic {top_topic_id + 1}** ({top_prob:.2%})")
                                                    
                                                    # Get the words from the primary topic
                                                    lda_topics = tm.get_lda_topics(num_words=10)
                                                    primary_topic_words = None
                                                    for tid, words in lda_topics:
                                                        if tid == top_topic_id:
                                                            primary_topic_words = words
                                                            break
                                                    
                                                    if primary_topic_words:
                                                        # Extract words from input text for matching
                                                        from narrativenexus_preprocess import clean_text, tokenize
                                                        cleaned_input = clean_text(test_text)
                                                        input_tokens = set(tokenize(cleaned_input))
                                                        
                                                        # Find matching words between input and topic
                                                        topic_words_set = {word for word, _ in primary_topic_words}
                                                        matching_words = input_tokens.intersection(topic_words_set)
                                                        
                                                        # Get top words from the topic
                                                        top_topic_words = [word for word, _ in primary_topic_words[:5]]
                                                        
                                                        # Create detailed explanation
                                                        st.markdown("---")
                                                        st.markdown("#### 💡 Why This Topic Was Selected")
                                                        
                                                        explanation_parts = []
                                                        
                                                        # Explain based on probability
                                                        if top_prob > 0.7:
                                                            confidence = "very high"
                                                        elif top_prob > 0.5:
                                                            confidence = "high"
                                                        elif top_prob > 0.3:
                                                            confidence = "moderate"
                                                        else:
                                                            confidence = "low"
                                                        
                                                        st.markdown(f"**Confidence Level:** {confidence.title()} ({top_prob:.1%})")
                                                        
                                                        # Show topic's key themes
                                                        st.markdown(f"**Topic {top_topic_id + 1} Key Themes:** {', '.join(top_topic_words)}")
                                                        
                                                        # Show matching words
                                                        if matching_words:
                                                            matching_list = list(matching_words)[:10]  # Show up to 10
                                                            st.markdown(f"**Matching Keywords Found:** {', '.join(matching_list)}")
                                                            st.info(f"✅ Your text contains **{len(matching_words)} words** that are strongly associated with Topic {top_topic_id + 1}, indicating a {top_prob:.1%} thematic alignment with this topic's core concepts.")
                                                        else:
                                                            st.markdown("**Matching Keywords:** Based on semantic similarity and statistical patterns")
                                                            st.info(f"📊 The model identified Topic {top_topic_id + 1} as the best match based on the overall semantic structure and word distribution patterns in your text, even though direct keyword matches may be limited.")
                                                        
                                                        # Show secondary topics if they exist
                                                        secondary_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)[1:3]
                                                        if secondary_topics and secondary_topics[0][1] > 0.1:
                                                            st.markdown("**Secondary Topics:**")
                                                            for tid, prob in secondary_topics:
                                                                if prob > 0.1:
                                                                    st.markdown(f"- Topic {tid + 1}: {float(prob):.1%} relevance")
                                                    
                                                except Exception as e:
                                                    status.update(label="❌ Analysis Failed", state="error", expanded=False)
                                                    st.error(f"Error analyzing text: {str(e)}")
                                
                                else:
                                    # NMF Processing
                                    st.info("⚡ Processing with NMF (Fast & Optimized)...")
                                    
                                    # Smart document splitting with size limit
                                    max_chars = 500000  # Process up to 500k characters
                                    if len(raw) > max_chars:
                                        st.warning(f"⚡ Large file detected ({len(raw):,} chars). Processing first {max_chars:,} characters.")
                                        raw_processed = raw[:max_chars]
                                    else:
                                        raw_processed = raw
                                    
                                    # Split into documents with better strategy
                                    para_splits = [p.strip() for p in raw_processed.split('\n\n') if len(p.strip()) > 30]
                                    
                                    if len(para_splits) >= num_topics * 2:
                                        docs = para_splits
                                    else:
                                        docs = [s.strip() for s in raw_processed.split('.') if len(s.strip()) > 20]
                                    
                                    # Limit documents for faster processing (max 1000 for NMF)
                                    if len(docs) > 1000:
                                        st.info(f"🚀 Sampling {1000} documents from {len(docs)} for faster training")
                                        import random
                                        random.seed(42)
                                        docs = random.sample(docs, 1000)
                                    
                                    if len(docs) < num_topics:
                                        st.warning(f"⚠️ Not enough documents ({len(docs)}) for {num_topics} topics.")
                                    else:
                                        # Progress indicator with status container
                                        with st.status(f"⚡ Training NMF model on {len(docs)} documents...", expanded=True) as status:
                                            st.write("📊 Vectorizing documents...")
                                            # Adaptive max_features based on document count
                                            max_features = min(1000, len(docs) * 5)
                                            
                                            st.write("🧮 Performing matrix factorization...")
                                            # Train NMF model
                                            tm.train_nmf_model(docs, num_topics=num_topics, max_features=max_features)
                                            
                                            status.update(label="✅ NMF Model Training Complete!", state="complete", expanded=False)
                                        
                                        st.success(f"✅ NMF model trained successfully!")
                                        st.markdown(f"**Processed:** {len(docs)} documents | **Features:** {max_features}")
                                        
                                        # Display topics
                                        st.markdown("#### 📊 Discovered Topics")
                                        topics = tm.get_nmf_topics(num_words=num_words)
                                        
                                        for topic_id, words in topics:
                                            word_list = ", ".join([f"{word} ({weight:.3f})" for word, weight in words[:5]])
                                            
                                            st.markdown(f"""
                                                <div style="
                                                    background: rgba(255, 255, 255, 0.12);
                                                    border-radius: 10px;
                                                    padding: 1rem;
                                                    margin: 0.5rem 0;
                                                    border-left: 4px solid #f093fb;
                                                ">
                                                    <h4 style="color: #ffffff; margin: 0 0 0.5rem 0;">🏷️ Topic {topic_id + 1}</h4>
                                                    <p style="color: #ffffff; margin: 0; font-size: 0.95rem;">{word_list}</p>
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                                            with st.expander(f"View all {num_words} words for Topic {topic_id + 1}"):
                                                for word, weight in words:
                                                    st.write(f"• {word}: {weight:.4f}")
                                        
                                        # Display statistics
                                        stats = tm.get_stats()
                                        st.markdown("#### 📊 Model Statistics")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("📄 Documents", f"{len(docs):,}")
                                        with col2:
                                            st.metric("🎯 Topics", f"{num_topics}")
                        
                        except Exception as e:
                            st.error(f"❌ Error during topic modeling: {str(e)}")
                            import traceback
                            with st.expander("View error details"):
                                st.code(traceback.format_exc())
                
                # Information boxes
                st.markdown("---")
                st.markdown("#### ℹ️ Algorithm Comparison")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **LDA (Recommended)**
                    - ✅ Probabilistic model
                    - ✅ Interpretable topics
                    - ✅ Handles overlapping themes
                    - ✅ Better for smaller datasets
                    - ⚠️ Slower training
                    """)
                
                with col2:
                    st.markdown("""
                    **NMF (Alternative)**
                    - ✅ Fast computation
                    - ✅ Sparse, distinct topics
                    - ✅ Good for large datasets
                    - ⚠️ Less interpretable
                    - ⚠️ Topics may be too separated
                    """)
    else:
        st.info("📭 No files available. Upload files in the Upload Files tab!")
