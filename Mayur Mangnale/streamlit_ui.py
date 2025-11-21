import streamlit as st
import pandas as pd

def set_page_config_and_style():
    st.set_page_config(
        page_title="AnalyzerNexus",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown("""
    <style>
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .stTextArea textarea {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #404040;
        }
        .stButton>button {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #606060;
        }
        .stButton>button:hover {
            background-color: #505050;
        }
        h1, h2, h3 {
            color: #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)


def show_metrics(data):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Words", data['total_words'])
    col2.metric("Sentiment", data['sentiment'])
    col3.metric("Topics", len(data['topics']))


def render_analysis_results(data):
    st.header("Results")
    show_metrics(data)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment", "Topics", "Summary", "Insights"])
    with tab1:
        st.write(f"**Sentiment:** {data['sentiment']}")
        st.write(f"**Score:** {data['sentiment_score']:.2%}")
        st.progress(data['sentiment_score'])

    with tab2:
        st.write("**Key Topics:**")
        for topic in data['topics']:
            st.write(f"- {topic['word']}: {topic['frequency']} mentions")

    with tab3:
        st.write("**Summary:**")
        for i, sent in enumerate(data['summary'], 1):
            st.write(f"{i}. {sent}")

    with tab4:
        st.write("**Insights:**")
        for insight in data['insights']:
            st.write(f"• {insight}")
        st.write("\n**Recommendations:**")
        for rec in data['recommendations']:
            st.write(f"• {rec}")

    st.divider()

    report = f"""
ANALYZERNEXUS - ANALYSIS REPORT
Generated: {data['timestamp']}
Source: {data['file_name']}

SUMMARY
Total Words: {data['total_words']}
Sentiment: {data['sentiment']} (Score: {data['sentiment_score']:.2%})

TOPICS
{chr(10).join([f"- {t['word']}: {t['frequency']} mentions" for t in data['topics']])}

SUMMARY
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(data['summary'])])}

INSIGHTS
{chr(10).join([f"- {i}" for i in data['insights']])}

RECOMMENDATIONS
{chr(10).join([f"- {r}" for r in data['recommendations']])}
"""
    st.download_button(
        "Download Report",
        data=report,
        file_name=f"report_{data['timestamp'].replace(':', '').replace(' ', '_')}.txt",
        mime="text/plain"
    )


def render_history(history):
    st.header("History")
    if history:
        st.dataframe(pd.DataFrame(history))
        if st.button("Clear History"):
            return True
    else:
        st.write("No analysis history yet.")
    return False
