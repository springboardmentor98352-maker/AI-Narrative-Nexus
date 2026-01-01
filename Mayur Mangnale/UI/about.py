import streamlit as st # type: ignore


def render_about():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class='card'>
                <h3>📖 About Narrative Nexus</h3>
            </div>
        """, unsafe_allow_html=True)
    
    # Main content - Refined and Enhanced
    st.markdown("""
        <div class='card'>
            <h2 style='color: #1e293b; margin-bottom: 1rem; font-size: 1.8rem;'>
                Intelligent Text Analysis at Your Fingertips
            </h2>
            <p style='font-size: 1.08rem; line-height: 1.9; color: #475569; margin-bottom: 1rem;'>
                <strong>Narrative Nexus</strong> is a cutting-edge AI-powered text analysis platform that transforms raw content 
                into actionable insights. Whether you're processing academic papers, analyzing customer feedback, mining research data, 
                or refining business reports, our sophisticated algorithms work behind the scenes to deliver clarity and understanding.
            </p>
            <p style='font-size: 1rem; line-height: 1.8; color: #64748b;'>
                Built with advanced Natural Language Processing (NLP) technology, Narrative Nexus eliminates the tedious manual work 
                of content analysis. In seconds, you'll uncover key themes, sentiment patterns, and linguistic insights that would 
                take hours to identify manually.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Mission Statement
    st.markdown("""
        <div style='background: linear-gradient(135deg, #6366f1, #8b5cf6); 
        padding: 2rem; border-radius: 16px; margin: 2rem 0; color: white;'>
            <h3 style='color: white; margin-top: 0;'>💡 Our Mission</h3>
            <p style='font-size: 1.05rem; line-height: 1.8; margin-bottom: 0;'>
                We believe every piece of text contains valuable insights waiting to be discovered. Our mission is to democratize 
                advanced text analysis, making enterprise-grade NLP tools accessible to everyone—from students to researchers to professionals.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Features section
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
            <div class='card'>
                <h3 style='color: #6366f1; margin-bottom: 1.5rem;'>⚡ Powerful Features</h3>
                <ul style='font-size: 0.98rem; color: #475569; line-height: 2.2;'>
                    <li><strong>🧠 Smart Summarization</strong><br><span style='font-size: 0.9rem; color: #64748b;'>AI-driven extractive summaries that capture the essence of your content</span></li>
                    <li><strong>💭 Sentiment Analysis</strong><br><span style='font-size: 0.9rem; color: #64748b;'>Detect emotional tone, positivity, negativity with precision scoring</span></li>
                    <li><strong>🔑 Keyword Extraction</strong><br><span style='font-size: 0.9rem; color: #64748b;'>Automatically identify the most important terms and concepts</span></li>
                    <li><strong>📊 Advanced Metrics</strong><br><span style='font-size: 0.9rem; color: #64748b;'>Word counts, readability scores, sentence analysis & lexical diversity</span></li>
                    <li><strong>📁 Universal Format Support</strong><br><span style='font-size: 0.9rem; color: #64748b;'>Seamlessly process TXT, PDF, and CSV files</span></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='card'>
                <h3 style='color: #8b5cf6; margin-bottom: 1.5rem;'>🏆 Why Narrative Nexus</h3>
                <ul style='font-size: 0.98rem; color: #475569; line-height: 2.2;'>
                    <li><strong>⚡ Lightning Fast</strong><br><span style='font-size: 0.9rem; color: #64748b;'>Process documents in milliseconds, not minutes</span></li>
                    <li><strong>🎨 Beautiful Experience</strong><br><span style='font-size: 0.9rem; color: #64748b;'>Intuitive interface with modern, responsive design</span></li>
                    <li><strong>🎯 Highly Accurate</strong><br><span style='font-size: 0.9rem; color: #64748b;'>Industry-leading NLP models for reliable results</span></li>
                    <li><strong>🔒 Your Privacy Matters</strong><br><span style='font-size: 0.9rem; color: #64748b;'>All processing happens locally—your data never leaves your machine</span></li>
                    <li><strong>💰 Completely Free</strong><br><span style='font-size: 0.9rem; color: #64748b;'>Open-source, no subscriptions, no hidden fees, forever free</span></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("""
        <div class='card'>
            <h3 style='color: #ec4899; margin-bottom: 1.5rem;'>⚙️ Simple 4-Step Workflow</h3>
        </div>
    """, unsafe_allow_html=True)
    
    step_cols = st.columns(4, gap="medium")
    steps = [
        ("📤", "Import Content", "Upload files or paste text directly into the editor", "#6366f1"),
        ("🔧", "Smart Processing", "Our algorithms clean, normalize, and prepare your data", "#8b5cf6"),
        ("🔍", "Deep Analysis", "Generate comprehensive summaries and extract key insights", "#d946ef"),
        ("📈", "Visualize Results", "Beautiful charts and metrics that tell your story", "#ec4899")
    ]
    
    for col, (emoji, title, desc, color) in zip(step_cols, steps):
        with col:
            st.markdown(f"""
                <div style='text-align: center; padding: 1.8rem; background: linear-gradient(135deg, #f8fafc, #f1f5f9); 
                border-radius: 14px; border: 2px solid #e2e8f0; transition: all 0.3s;'>
                    <div style='font-size: 2.5rem; margin-bottom: 0.8rem;'>{emoji}</div>
                    <h4 style='color: {color}; margin-bottom: 0.8rem; font-size: 1.05rem;'>{title}</h4>
                    <p style='font-size: 0.92rem; color: #64748b; line-height: 1.6;'>{desc}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Use Cases
    st.markdown("""
        <hr style='margin: 2.5rem 0; border-color: #e2e8f0;'>
        <div class='card'>
            <h3 style='color: #10b981; margin-bottom: 1.5rem;'>🎯 Perfect For</h3>
        </div>
    """, unsafe_allow_html=True)
    
    use_case_cols = st.columns(3, gap="large")
    use_cases = [
        ("👨‍🎓", "Students & Academics", "Analyze research papers, essays, and academic texts quickly"),
        ("💼", "Business Professionals", "Process reports, proposals, and market research efficiently"),
        ("📊", "Data Scientists", "Preprocess and analyze large text datasets for ML projects")
    ]
    
    for col, (icon, title, desc) in zip(use_case_cols, use_cases):
        with col:
            st.markdown(f"""
                <div style='padding: 1.5rem; background: #f8fafc; border-radius: 12px; 
                border-left: 4px solid #6366f1;'>
                    <h4 style='color: #1e293b; margin: 0 0 0.5rem 0;'>{icon} {title}</h4>
                    <p style='color: #64748b; margin: 0; font-size: 0.95rem; line-height: 1.6;'>{desc}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer section
    st.markdown("<hr style='margin: 3rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f0f9ff, #f5f3ff); 
        border-radius: 16px; border: 2px solid #e0e7ff;'>
            <h2 style='color: #1e293b; margin: 0 0 1rem 0; font-size: 1.5rem;'>Ready to Transform Your Text?</h2>
            <p style='font-size: 1rem; color: #475569; margin: 0 0 1.5rem 0; line-height: 1.6;'>
                Get started in seconds. Head to the <strong>📝 Text Input</strong> tab to upload your first document, 
                or explore the <strong>🔍 Analysis</strong> dashboard to see what's possible.
            </p>
            <p style='font-size: 0.9rem; color: #64748b; margin: 0;'>
                No setup required. No registration. Just brilliant text analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)
