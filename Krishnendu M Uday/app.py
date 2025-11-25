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

tabs = st.tabs(["📤 Upload Files", "📊 File Analysis", "🔬 Text Processing"])

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
