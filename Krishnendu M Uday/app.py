import streamlit as st
import os
from narrativenexus_utils import (
    ensure_data_dir,
    save_uploaded_file,
    get_sample_files,
    parse_preview,
    read_full_text,
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
    [data-testid="stMetricValue"] {
        color: #667eea;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
        border-radius: 10px;
        border-left: 4px solid #667eea;
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
    h2, h3 {
        color: #667eea !important;
        font-weight: 600 !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("NarrativeNexus")

tabs = st.tabs(["Day 1 - Upload", "Day 2 - Analysis", "Day 3 - Advanced"])

# Day 1: Upload
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

# Day 2: Preview & Word Count
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
        st.info("📭 No files available. Upload files in Day 1!")

# Day 3: Advanced Analysis
with tabs[2]:
    st.markdown("### 🔬 Advanced Text Analysis")
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
                        
                        cleaned = clean_text(raw)
                        toks = tokenize(cleaned)

                        st.markdown("#### 🧹 Cleaned Text Preview")
                        st.code(cleaned[:2000] + ("..." if len(cleaned) > 2000 else ""), language=None)

                        st.markdown("#### 📊 Token Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🔢 Total Tokens", f"{len(toks):,}")
                        with col2:
                            st.metric("🔣 Unique Tokens", f"{len(set(toks)):,}")

                        from collections import Counter
                        freq = Counter([t.lower() for t in toks])
                        top = freq.most_common(20)
                        
                        st.markdown("#### 🏆 Top 20 Frequent Tokens")
                        if top:
                            # Create a more visual representation
                            for i, (token, count) in enumerate(top, 1):
                                bar_length = int((count / top[0][1]) * 30)
                                bar = "█" * bar_length
                                st.markdown(f"`{i:2d}` **{token}** {'`' + bar + '`'} {count}")
                        else:
                            st.write("No tokens found.")

                        if SPACY_AVAILABLE:
                            st.info("✓ Using spaCy for advanced tokenization & lemmatization")
                        else:
                            st.warning("⚠ Using fallback preprocessing (spaCy not available)")
    else:
        st.info("📭 No files available. Upload files in Day 1!")

