import streamlit as st # type: ignore
import pandas as pd # type: ignore
from io import BytesIO
from datetime import datetime
from metrics import (
    word_count, sentence_count, sentiment_analysis,
    sentiment_distribution, sentiment_to_emoji,
    top_tokens, simple_summary, extract_topics, readability_score,
    comprehensive_summary
)

def generate_text_report(text, sentiment_scores, tokens, summary):
    """Generate report content for text data"""
    wc = word_count(text)
    sc = sentence_count(text)
    compound_score = sentiment_scores["compound"]
    
    if compound_score > 0.2:
        sentiment_text = "POSITIVE"
    elif compound_score < -0.2:
        sentiment_text = "NEGATIVE"
    else:
        sentiment_text = "NEUTRAL"
    
    distribution = sentiment_distribution(sentiment_scores)
    
    report = f"""
═══════════════════════════════════════════════════════════════
                    NARRATIVE NEXUS - TEXT ANALYSIS REPORT
═══════════════════════════════════════════════════════════════

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

─────────────────────────────────────────────────────────────────
📊 KEY METRICS
─────────────────────────────────────────────────────────────────
• Total Words:              {wc}
• Total Sentences:          {sc}
• Average Word Length:      {round(len(text) / max(wc, 1), 2)}
• Character Count:          {len(text):,}

─────────────────────────────────────────────────────────────────
💭 SENTIMENT ANALYSIS
─────────────────────────────────────────────────────────────────
• Overall Sentiment:        {sentiment_text}
• Compound Score:           {compound_score:.4f}
• Positive Sentences:       {distribution.get('Positive', 0) * 100:.1f}%
• Neutral Sentences:        {distribution.get('Neutral', 0) * 100:.1f}%
• Negative Sentences:       {distribution.get('Negative', 0) * 100:.1f}%

─────────────────────────────────────────────────────────────────
🔑 TOP KEYWORDS
─────────────────────────────────────────────────────────────────
"""
    
    for i, (token, count) in enumerate(tokens, 1):
        report += f"{i:2d}. {token.upper():<20} (frequency: {count})\n"
    
    report += f"""
─────────────────────────────────────────────────────────────────
📝 SUMMARY
─────────────────────────────────────────────────────────────────
{summary}

═══════════════════════════════════════════════════════════════
                        END OF REPORT
═══════════════════════════════════════════════════════════════
"""
    return report

def render_analysis():
    # Check for data in session state
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(217, 119, 6, 0.1));
            backdrop-filter: blur(10px); border-left: 4px solid #f59e0b; padding: 1.8rem; 
            border-radius: 16px; margin: 2rem 0;'>
                <p style='color: #92400e; margin: 0; font-weight: 700; font-size: 1.05rem;'>
                    ⚠️ No data to analyze. Please upload text in the <strong>Upload</strong> section first.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    text = st.session_state.processed_data
    source_type = st.session_state.data_type

    # ==================== TEXT ANALYSIS ====================
    if source_type == "text":
        # Calculate metrics
        wc = word_count(text)
        sc = sentence_count(text)
        sentiment_scores = sentiment_analysis(text)
        sentiment = sentiment_to_emoji(sentiment_scores["compound"])
        distribution = sentiment_distribution(sentiment_scores)
        tokens = top_tokens(text, n=12)
        summary = comprehensive_summary(text, sentiment_scores, tokens)

        # ==================== KEY METRICS ====================
        metric_cols = st.columns(3, gap="large")
        
        metrics_data = [
            ("📝", "Words", wc, "#6366f1"),
            ("📚", "Sentences", sc, "#8b5cf6"),
            ("⏱️", "Avg Length", f"{round(len(text) / max(wc, 1), 1)}", "#d946ef")
        ]
        
        for col, (icon, label, value, color) in zip(metric_cols, metrics_data):
            with col:
                st.markdown(f"""
                    <div class='metric-card' style='border-left: 4px solid {color};'>
                        <div class='metric-label'>{icon} {label}</div>
                        <div class='metric-value'>{value}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

        # ==================== TOP TOKENS ====================
        st.markdown("""
            <h3 style='color: #8b5cf6; margin-bottom: 1.5rem; font-size: 1.5rem; font-weight: 800;'>🔑 Key Terms</h3>
        """, unsafe_allow_html=True)
        
        # Simple table view instead of styled chips
        token_data = []
        for i, (tok, cnt) in enumerate(tokens):
            token_data.append({"Term": tok.upper(), "Frequency": cnt})
        
        st.dataframe(
            pd.DataFrame(token_data),
            use_container_width=True,
            hide_index=True,
            height=300
        )


        # ==================== SENTIMENT ANALYSIS ====================
        st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
        
        sentiment_cols = st.columns(2, gap="large")
        
        with sentiment_cols[0]:
            compound_score = sentiment_scores["compound"]
            if compound_score > 0.2:
                sentiment_color = "#10b981"
                sentiment_text = "POSITIVE"
            elif compound_score < -0.2:
                sentiment_color = "#ef4444"
                sentiment_text = "NEGATIVE"
            else:
                sentiment_color = "#f59e0b"
                sentiment_text = "NEUTRAL"
            
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
                backdrop-filter: blur(20px); padding: 3rem 2rem; border-radius: 24px;
                border: 1.5px solid rgba(255, 255, 255, 0.8); box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
                text-align: center;'>
                    <p style='color: #64748b; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; margin: 0 0 1rem 0;'>
                        Sentiment Analysis
                    </p>
                    <div style='font-size: 4rem; margin: 1rem 0; animation: bounce 2s infinite;'>
                        {sentiment}
                    </div>
                    <h2 style='color: {sentiment_color}; margin: 1rem 0; font-weight: 950; font-size: 2rem;'>
                        {sentiment_text}
                    </h2>
                    <p style='color: #64748b; margin: 0; font-size: 1.1rem; font-weight: 700;'>
                        Score: <span style='color: {sentiment_color};'>{compound_score:.3f}</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with sentiment_cols[1]:
            st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
                backdrop-filter: blur(20px); padding: 2.5rem; border-radius: 24px;
                border: 1.5px solid rgba(255, 255, 255, 0.8); box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);'>
                    <p style='color: #64748b; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; margin: 0 0 1.5rem 0;'>
                        Distribution
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            sentiment_data = {
                "Positive": distribution.get("Positive", 0) * 100,
                "Neutral": distribution.get("Neutral", 0) * 100,
                "Negative": distribution.get("Negative", 0) * 100
            }
            
            for label, value in sentiment_data.items():
                if label == "Positive":
                    color = "#10b981"
                elif label == "Negative":
                    color = "#ef4444"
                else:
                    color = "#f59e0b"
                
                st.markdown(f"""
                    <div style='margin-bottom: 1.5rem;'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 0.8rem;'>
                            <span style='font-weight: 700; color: #1e293b;'>{label}</span>
                            <span style='font-weight: 800; color: {color}; font-size: 1.1rem;'>{value:.1f}%</span>
                        </div>
                        <div style='width: 100%; height: 12px; background: linear-gradient(90deg, rgba(0,0,0,0.05), rgba(0,0,0,0.1)); 
                        border-radius: 8px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);'>
                            <div style='width: {value}%; height: 100%; background: linear-gradient(90deg, {color}, {color}dd); 
                            border-radius: 8px; transition: width 0.6s cubic-bezier(0.23, 1, 0.320, 1);'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # ==================== TOPIC MODELING ====================
        st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
            <h3 style='color: #8b5cf6; margin-bottom: 1.5rem; font-size: 1.5rem; font-weight: 800;'>🎯 Main Topics</h3>
        """, unsafe_allow_html=True)
        
        try:
            topics = extract_topics(text, n_topics=3)
            
            topic_cols = st.columns(3, gap="large")
            
            for idx, (col, (topic_name, keywords)) in enumerate(zip(topic_cols, topics.items())):
                with col:
                    colors = ["#6366f1", "#8b5cf6", "#d946ef"]
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
                        backdrop-filter: blur(20px); padding: 2rem; border-radius: 20px;
                        border-left: 4px solid {colors[idx]};
                        border: 1.5px solid rgba(255, 255, 255, 0.8); 
                        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
                        text-align: center;'>
                            <p style='color: {colors[idx]}; font-weight: 800; margin: 0 0 1rem 0; font-size: 1.2rem;'>
                                {topic_name}
                            </p>
                            <div style='color: #475569; line-height: 2; font-size: 0.95rem;'>
                                {', '.join(keywords)}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.warning("⚠️ Topics could not be extracted. Text may be too short.")

        # ==================== READABILITY SCORE ====================
        st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
            <h3 style='color: #0ea5e9; margin-bottom: 1.5rem; font-size: 1.5rem; font-weight: 800;'>📚 Readability Analysis</h3>
        """, unsafe_allow_html=True)
        
        try:
            readability = readability_score(text)
            
            # Interpret readability score
            if readability < 6:
                level = "Elementary School"
                color = "#10b981"
            elif readability < 9:
                level = "Middle School"
                color = "#0ea5e9"
            elif readability < 13:
                level = "High School"
                color = "#f59e0b"
            elif readability < 16:
                level = "College"
                color = "#d946ef"
            else:
                level = "Graduate"
                color = "#ef4444"
            
            read_cols = st.columns(2, gap="large")
            
            with read_cols[0]:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
                    backdrop-filter: blur(20px); padding: 3rem 2rem; border-radius: 24px;
                    border: 1.5px solid rgba(255, 255, 255, 0.8); 
                    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
                    text-align: center;'>
                        <p style='color: #64748b; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; margin: 0 0 1rem 0;'>
                            Grade Level
                        </p>
                        <div style='font-size: 3.5rem; margin: 1rem 0; font-weight: 950; color: {color};'>
                            {readability:.1f}
                        </div>
                        <p style='color: {color}; margin: 1rem 0; font-size: 1.2rem; font-weight: 800;'>
                            {level}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with read_cols[1]:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
                    backdrop-filter: blur(20px); padding: 2.5rem; border-radius: 24px;
                    border: 1.5px solid rgba(255, 255, 255, 0.8); 
                    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);'>
                        <p style='color: #64748b; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; margin: 0 0 1.5rem 0;'>
                            Interpretation
                        </p>
                        <div style='color: #475569; line-height: 1.8; font-size: 1rem;'>
                            <p style='margin: 0.5rem 0;'><strong>Score:</strong> {readability:.1f}</p>
                            <p style='margin: 0.5rem 0;'><strong>Level:</strong> {level}</p>
                            <p style='margin: 0.5rem 0; font-size: 0.95rem; color: #64748b;'>
                                The text is written at a level suitable for <strong>{level.lower()}</strong> readers.
                            </p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.warning("⚠️ Readability score could not be calculated. Text may be too short.")

        # ==================== SUMMARY ====================
        st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
            backdrop-filter: blur(20px); padding: 2.5rem; border-radius: 24px;
            border: 1.5px solid rgba(255, 255, 255, 0.8); box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
            margin-bottom: 2rem;'>
                <h3 style='color: #10b981; margin: 0 0 1.5rem 0; font-size: 1.5rem; font-weight: 800;'>
                    📝 Summary
                </h3>
                <p style='color: #475569; line-height: 1.9; margin: 0; font-size: 1.05rem;'>
                    {summary}
                </p>
            </div>
        """, unsafe_allow_html=True)

        # ==================== DOWNLOAD REPORTS ====================
        st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
            backdrop-filter: blur(20px); padding: 2.5rem; border-radius: 24px;
            border: 1.5px solid rgba(255, 255, 255, 0.8); box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
            margin-bottom: 2rem;'>
                <h3 style='color: #3b82f6; margin: 0 0 1.5rem 0; font-size: 1.5rem; font-weight: 800;'>
                    📥 Download Report
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Generate report content
        report_text = generate_text_report(text, sentiment_scores, tokens, summary)
        
        st.download_button(
            label="📄 Download Full Report (TXT)",
            data=report_text,
            file_name=f"narrative_nexus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_txt"
        )

    # ==================== CSV DATA ====================
    else:
        st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
            backdrop-filter: blur(20px); padding: 2.5rem; border-radius: 24px;
            border: 1.5px solid rgba(255, 255, 255, 0.8); box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
            margin-bottom: 2rem;'>
                <h3 style='color: #6366f1; margin: 0 0 1.5rem 0; font-size: 1.5rem; font-weight: 800;'>
                    📊 Data Preview
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(text.head(10), use_container_width=True, height=400)
        
        stat_cols = st.columns(3, gap="large")
        
        stats = [
            ("📋", "Rows", len(text), "#6366f1"),
            ("📊", "Columns", len(text.columns), "#8b5cf6"),
            ("🔢", "Cells", text.shape[0] * text.shape[1], "#d946ef")
        ]
        
        for col, (icon, label, value, color) in zip(stat_cols, stats):
            with col:
                st.markdown(f"""
                    <div class='metric-card' style='border-left: 4px solid {color};'>
                        <div class='metric-label'>{icon} {label}</div>
                        <div class='metric-value'>{value}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # ==================== CSV DOWNLOAD ====================
        st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
            backdrop-filter: blur(20px); padding: 2.5rem; border-radius: 24px;
            border: 1.5px solid rgba(255, 255, 255, 0.8); box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
            margin-bottom: 2rem;'>
                <h3 style='color: #3b82f6; margin: 0 0 1.5rem 0; font-size: 1.5rem; font-weight: 800;'>
                    📥 Download Data
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            csv_buffer = BytesIO()
            text.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="📊 Download as CSV",
                data=csv_buffer,
                file_name=f"narrative_nexus_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv_data"
            )
        
        with col2:
            # Generate JSON from CSV
            json_str = text.to_json(orient='records', indent=2)
            
            st.download_button(
                label="📋 Download as JSON",
                data=json_str,
                file_name=f"narrative_nexus_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                key="download_json_data"
            )
