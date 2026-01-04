import streamlit as st # type: ignore
from UI.layout import render_header, render_menu
from UI.about import render_about
from UI.text_input import render_text_input
from UI.analysis import render_analysis
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Narrative Nexus - Text Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== RENDER HEADER ====================
render_header()

# ==================== NAVIGATION ====================
selected = render_menu()

# ==================== ROUTE HANDLER ====================
if selected == "Home":
    # Hero Section
    st.markdown(
        "<div style='text-align: center; padding: 4rem 2rem 3rem;'><div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 4rem 3rem; border-radius: 32px; box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3); position: relative; overflow: hidden;'><div style='position: relative; z-index: 1;'><h1 style='color: #ffffff; font-size: 3.2rem; margin: 0 0 1rem 0; font-weight: 900; letter-spacing: -1px;'>Narrative Nexus</h1><p style='color: rgba(255, 255, 255, 0.95); font-size: 1.3rem; margin: 0 0 2rem 0; line-height: 1.8; font-weight: 500;'>Unlock Deep Insights from Your Text with Advanced AI Analysis</p><p style='color: rgba(255, 255, 255, 0.85); font-size: 1.05rem; margin: 0; line-height: 1.6;'>Analyze sentiment, extract key topics, measure readability, and uncover patterns in your content</p></div></div></div>",
        unsafe_allow_html=True
    )
    
    # Features Section
    st.markdown("<div style='margin-top: 4rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <h2 style='text-align: center; color: #1e293b; font-size: 2rem; margin: 0 0 3rem 0; font-weight: 800;'>
            What You Can Do
        </h2>
    """, unsafe_allow_html=True)
    
    feature_cols = st.columns(4, gap="large")
    features = [
        {
            "emoji": "📤",
            "title": "Smart Upload",
            "desc": "Upload TXT, PDF, or CSV files with automatic detection",
            "color": "#6366f1"
        },
        {
            "emoji": "🧠",
            "title": "Sentiment Analysis",
            "desc": "Detect emotional tone and sentiment distribution",
            "color": "#8b5cf6"
        },
        {
            "emoji": "🔑",
            "title": "Key Extraction",
            "desc": "Identify top keywords and topics automatically",
            "color": "#d946ef"
        },
        {
            "emoji": "📊",
            "title": "Rich Metrics",
            "desc": "Get readability scores, word counts, and more",
            "color": "#0ea5e9"
        }
    ]
    
    for col, feature in zip(feature_cols, features):
        with col:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.9));
                backdrop-filter: blur(20px); padding: 2.5rem 1.8rem; border-radius: 24px; 
                border: 1.5px solid rgba(255, 255, 255, 0.8); box-shadow: 0 12px 32px rgba(0, 0, 0, 0.06);
                transition: all 0.3s ease; text-align: center; height: 100%;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>{feature["emoji"]}</div>
                    <h4 style='color: {feature["color"]}; margin: 0 0 0.8rem 0; font-size: 1.2rem; font-weight: 800;'>
                        {feature["title"]}
                    </h4>
                    <p style='color: #64748b; margin: 0; font-size: 0.95rem; line-height: 1.6;'>
                        {feature["desc"]}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("<div style='margin-top: 4rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; padding: 3rem 2rem;'>
            <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 197, 94, 0.1));
            padding: 2.5rem; border-radius: 24px; border: 2px solid rgba(16, 185, 129, 0.3);'>
                <h3 style='color: #059669; margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 800;'>
                    Ready to Analyze?
                </h3>
                <p style='color: #047857; margin: 0; font-size: 1.05rem;'>
                    👉 Head to the <strong>Upload</strong> section to get started!
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif selected == "Upload":
    render_text_input()

elif selected == "Analytics":
    render_analysis()

# ==================== FOOTER ====================
st.markdown("""
    <div style='margin-top: 4rem; padding: 2.5rem; text-align: center; border-top: 1.5px solid rgba(99, 102, 241, 0.1);'>
        <p style='color: #64748b; font-size: 0.95rem; margin: 0;'>
            <strong>Narrative Nexus</strong> — Advanced Text Analysis Platform
        </p>
    </div>
""", unsafe_allow_html=True)

