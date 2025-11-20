import streamlit as st
from datetime import datetime

from preprocessing import (
    clean_text,
    remove_stopwords,
    tokenize,
    analyze_sentiment,
    extract_topics,
    extractive_summarization,
    generate_insights
)
from file_parse import parse_file
import streamlit_ui as ui

def main():
    ui.set_page_config_and_style()

    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'history' not in st.session_state:
        st.session_state.history = []

    st.title("AnalyzerNexus")
    st.write("Advanced Text Analysis Platform")
    st.divider()
    st.header("Upload or Enter Text")

    uploaded_file = st.file_uploader("Choose file (.txt, .csv, .docx, .pdf)", type=['txt', 'csv', 'docx', 'pdf'])
    text_input = ""
    file_name = "Direct Input"

    if uploaded_file:
        text_input = parse_file(uploaded_file)
        file_name = uploaded_file.name
        st.success(f"File loaded: {file_name}")
    else:
        text_input = st.text_area("Or paste text here", height=150)

    if st.button("Analyze"):
        if not text_input or len(text_input.strip()) < 10:
            st.error("Please provide valid text")
        else:
            with st.spinner("Analyzing..."):
                cleaned_text = clean_text(text_input)
                processed_text = remove_stopwords(cleaned_text)
                tokens = tokenize(processed_text)
                sentiment, sentiment_score = analyze_sentiment(processed_text)
                topics = extract_topics(processed_text, 3)
                summary = extractive_summarization(text_input, 2)
                insights, recommendations = generate_insights(sentiment, topics, len(tokens))
                st.session_state.analysis_data = {
                    'file_name': file_name,
                    'sentiment': sentiment,
                    'sentiment_score': sentiment_score,
                    'topics': topics,
                    'summary': summary,
                    'insights': insights,
                    'recommendations': recommendations,
                    'total_words': len(tokens),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.session_state.history.append({
                    'timestamp': st.session_state.analysis_data['timestamp'],
                    'file': file_name,
                    'words': len(tokens),
                    'sentiment': sentiment
                })
            st.success("Analysis Complete!")

    st.divider()

    if st.session_state.analysis_data:
        ui.render_analysis_results(st.session_state.analysis_data)

    if ui.render_history(st.session_state.history):
        st.session_state.history = []
        st.experimental_rerun()


if __name__ == "__main__":
    main()
