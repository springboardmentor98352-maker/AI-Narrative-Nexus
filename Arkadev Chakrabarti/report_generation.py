import streamlit as st
import pandas as pd
from docx import Document
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from heapq import nlargest
import numpy as np
import base64
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS

# --- Comprehensive Report ---
                st.subheader("📑 Comprehensive Report")

                from datetime import datetime
                current_timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

                overall_sent = analyze_sentiment(st.session_state.text_data)['category']

                # Default values if topic modeling hasn't been run
                themes_text = "No topics identified yet (run Topic Modeling to see themes)."
                dominant_topics_desc = "Not available — please run Topic Modeling first."

                if st.session_state.topics_data and st.session_state.topic_prevalence is not None:
                    # Build full themes list
                    themes_lines = []
                    for idx, (words, _, _, _) in enumerate(st.session_state.topics_data):
                        word_list = normalize_topic_words(words)
                        themes_lines.append(f"**Topic {idx + 1}:** {', '.join(word_list[:8])}")
                    themes_text = "\n".join(themes_lines)

                    # Get top 3 dominant topics with their representative keywords
                    prevalence = st.session_state.topic_prevalence
                    top_indices = np.argsort(prevalence)[-3:][::-1]  # Highest to lowest

                    dominant_parts = []
                    for rank, topic_idx in enumerate(top_indices, 1):
                        words = normalize_topic_words(st.session_state.topics_data[topic_idx][1])
                        top_keywords = ', '.join(words[:5])  # Top 5 keywords for brevity
                        percentage = prevalence[topic_idx] * 100
                        dominant_parts.append(f"{rank}. **Topic {topic_idx + 1}** ({percentage:.1f}% prevalence): {top_keywords}")

                    dominant_topics_desc = "\n".join(dominant_parts)

                # Final report with timestamp and richer content
                report = f"""
**Narrative Nexus – Text Analysis Report**  
*Generated on {current_timestamp}*

**Document Overview**
- Total sentences: {len(sent_tokenize(st.session_state.text_data))}
- Total tokens (after cleaning): {len(st.session_state.tokens) if st.session_state.tokens else 'N/A'}

**Overall Sentiment**  
**{overall_sent}** (Polarity: {analyze_sentiment(st.session_state.text_data)['polarity']:.3f})

**Key Themes Identified**
{themes_text}

**Top 3 Dominant Themes**
{dominant_topics_desc}

**Insights & Actionable Recommendations**
- The primary focus of the text revolves around the dominant themes listed above.
- Sentiment is **{overall_sent.lower()}**, suggesting {'strong positive engagement and approval' if overall_sent == 'Positive' else 'areas of concern that may require attention or improvement' if overall_sent == 'Negative' else 'a balanced or neutral viewpoint with room for deeper interpretation'}.
- {'Prioritize communication, content, or strategy around the top dominant themes to maximize impact and resonance.' if st.session_state.topics_data else 'Run Topic Modeling to unlock specific theme-based recommendations.'}
- For longitudinal tracking, consider analyzing multiple documents over time to monitor shifts in themes and sentiment.
"""

                st.markdown(report)
