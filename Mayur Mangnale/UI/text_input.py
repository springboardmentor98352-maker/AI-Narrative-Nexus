import streamlit as st # type: ignore
from data_extractor import extract_text_from_file
from data_preprocessing import preprocess_text
import pandas as pd # type: ignore

def render_text_input():
    """Clean, single upload and text input interface with session state"""
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'data_type' not in st.session_state:
        st.session_state.data_type = None
    
    # Title
    st.markdown("""
        <h2 style='text-align: center; color: #6366f1; margin-bottom: 2rem; font-weight: 900;'>
            📨 Upload & Analyze
        </h2>
    """, unsafe_allow_html=True)
    
    # Two column layout
    col_upload, col_text = st.columns(2, gap="large")
    
    with col_upload:
        st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
            backdrop-filter: blur(20px); padding: 2.5rem; border-radius: 24px;
            border: 1.5px solid rgba(255, 255, 255, 0.8); box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);'>
                <p style='color: #64748b; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; 
                letter-spacing: 1px; margin: 0 0 1.5rem 0;'>📤 Upload File</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a file (TXT, PDF, or CSV)",
            type=["txt", "csv", "pdf"],
            label_visibility="collapsed",
            key="file_uploader_main"
        )
        
        if uploaded_file:
            file_size = uploaded_file.size / 1024
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
                backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 16px; margin-top: 1.5rem;
                border-left: 4px solid #10b981;'>
                    <p style='margin: 0; color: #065f46; font-weight: 700;'>✓ {uploaded_file.name}</p>
                    <p style='margin: 0.5rem 0 0 0; color: #065f46; font-size: 0.9rem;'>{file_size:.2f} KB</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col_text:
        st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
            backdrop-filter: blur(20px); padding: 2.5rem; border-radius: 24px;
            border: 1.5px solid rgba(255, 255, 255, 0.8); box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);'>
                <p style='color: #64748b; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; 
                letter-spacing: 1px; margin: 0 0 1.5rem 0;'>✍️ Or Paste Text</p>
            </div>
        """, unsafe_allow_html=True)
        
        pasted_text = st.text_area(
            "Paste your text here...",
            height=250,
            label_visibility="collapsed",
            placeholder="Enter or paste your text content here...",
            key="text_area_main"
        )
    
    # Divider
    st.markdown("""
        <div style='height: 2px; background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent); 
        margin: 3rem 0; border: none;'></div>
    """, unsafe_allow_html=True)
    
    # Analyze Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button(
            "🚀 Analyze Now",
            use_container_width=True,
            key="analyze_button"
        )
    
    if analyze_button:
        with st.spinner("✨ Processing your content..."):
            # Extract text
            raw_text, file_type, df_data, error = extract_text_from_file(
                uploaded_file=uploaded_file,
                pasted_text=pasted_text
            )
            
            if error:
                st.error(f"❌ {error}")
                return
            
            # Show success
            st.success(f"✅ {file_type.upper()} content loaded successfully!")
            
            # Process based on file type
            if file_type in ["txt", "pdf"]:
                # Preprocess
                processed, err = preprocess_text(raw_text, file_type)
                if err:
                    st.error(f"❌ {err}")
                    return
                
                # Store in session state
                st.session_state.processed_data = processed
                st.session_state.data_type = "text"
                
                # Show preview
                preview_text = processed[:1500] + "..." if len(processed) > 1500 else processed
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
                    backdrop-filter: blur(20px); padding: 2.5rem; border-radius: 24px;
                    border: 1.5px solid rgba(99, 102, 241, 0.2); box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
                    margin-top: 2rem;'>
                        <h3 style='color: #6366f1; margin: 0 0 1.5rem 0; font-weight: 800;'>📋 Preview</h3>
                        <p style='color: #475569; line-height: 1.8; margin: 0; font-size: 0.95rem;'>
                            {preview_text}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Stats
                col1, col2, col3 = st.columns(3, gap="large")
                stats = [
                    ("📝", "Words", len(processed.split()), "#6366f1"),
                    ("🔤", "Characters", len(processed), "#8b5cf6"),
                    ("📚", "Sentences", len([s for s in processed.split('.') if s.strip()]), "#d946ef")
                ]
                
                for col, (icon, label, value, color) in zip([col1, col2, col3], stats):
                    with col:
                        st.markdown(f"""
                            <div class='metric-card' style='border-left: 4px solid {color}; margin-top: 2rem;'>
                                <div class='metric-label'>{icon} {label}</div>
                                <div class='metric-value'>{value}</div>
                            </div>
                        """, unsafe_allow_html=True)
            
            elif file_type == "csv":
                # Preprocess
                processed_df, err = preprocess_text(text=None, file_type="csv", df=df_data)
                if err:
                    st.error(f"❌ {err}")
                    return
                
                # Store in session state
                st.session_state.processed_data = processed_df
                st.session_state.data_type = "csv"
                
                st.markdown("""
                    <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
                    backdrop-filter: blur(20px); padding: 2.5rem; border-radius: 24px;
                    border: 1.5px solid rgba(99, 102, 241, 0.2); box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
                    margin-top: 2rem;'>
                        <h3 style='color: #6366f1; margin: 0 0 1.5rem 0; font-weight: 800;'>📊 Preview</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(processed_df.head(), use_container_width=True)
        
        # Success message
        st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(3, 102, 214, 0.1), rgba(12, 74, 110, 0.1));
            backdrop-filter: blur(10px); border-left: 4px solid #0284c7; padding: 1.8rem; 
            border-radius: 16px; margin-top: 2rem;'>
                <p style='color: #0c4a6e; margin: 0; font-weight: 700; font-size: 1.05rem;'>
                    ✨ Ready! Head to <strong>Analytics</strong> to explore insights & download reports.
                </p>
            </div>
        """, unsafe_allow_html=True)
