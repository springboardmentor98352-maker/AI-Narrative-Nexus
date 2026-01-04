import streamlit as st # type: ignore
from streamlit_option_menu import option_menu # type: ignore
import os


def render_header():
    st.set_page_config(
        page_title="Narrative Nexus",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load CSS
    css_path = os.path.join("Styles", "main.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Centered Header section with immersive design
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0 1rem 0;'>
            <div class='center-text'>
                <h1 style='margin: 0; padding: 0;'>Narrative Nexus</h1>
                <h3 style='margin: 0.5rem 0 0 0;'>Text Analysis Reimagined</h3>
                <p style='margin-top: 1rem;'>✨ Analyze • Summarize • Understand ✨</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='height: 2px; background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent); 
        margin: 2rem 0; border: none;'></div>
    """, unsafe_allow_html=True)


def render_menu():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Upload", "Analytics"],
        icons=["house", "cloud-upload", "graph-up"],
        default_index=1,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0px",
                "background-color": "transparent",
                "display": "flex",
                "gap": "1rem",
                "justify-content": "center",
                "flex-wrap": "wrap",
                "margin-bottom": "2rem"
            },
            "icon": {
                "color": "#6366f1",
                "font-size": "1.3rem"
            },
            "nav-link": {
                "font-size": "1.05rem",
                "text-align": "center",
                "margin": "0px",
                "padding": "14px 28px",
                "border-radius": "14px",
                "background": "rgba(255, 255, 255, 0.7)",
                "backdrop-filter": "blur(10px)",
                "border": "1.5px solid rgba(255, 255, 255, 0.5)",
                "color": "#64748b",
                "transition": "all 0.4s cubic-bezier(0.23, 1, 0.320, 1)",
                "font-weight": "700",
                "letter-spacing": "0.5px"
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
                "color": "white",
                "border": "1.5px solid rgba(99, 102, 241, 0.5)",
                "box-shadow": "0 10px 40px rgba(99, 102, 241, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.3)",
            },
        }
    )
    
    return selected.split()[0] if selected else "Home"

