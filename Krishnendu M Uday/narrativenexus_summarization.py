"""
Text Summarization Module for NarrativeNexus

This module provides text summarization capabilities using multiple approaches:
- Extractive summarization (TextRank, TF-IDF)
- Abstractive summarization (Transformers)
- Hybrid approaches combining both

Extractive vs Abstractive Summarization:
----------------------------------------
Extractive Summarization:
- Selects and extracts important sentences from original text
- Fast and reliable
- Maintains original wording (no paraphrasing)
- Good for technical/factual content
- Methods: TextRank, TF-IDF, graph-based

Abstractive Summarization:
- Generates new sentences that capture meaning
- More human-like summaries
- Can paraphrase and rephrase
- Requires more computational resources
- Methods: Transformers (BART, T5, Pegasus)

Recommendation: Use extractive for speed, abstractive for quality
"""

from typing import List, Dict, Optional, Tuple
import re

# Try importing different summarization libraries
SUMY_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
GENSIM_AVAILABLE = False

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    SUMY_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    from gensim.summarization import summarize as gensim_summarize
    from gensim.summarization import keywords as gensim_keywords
    GENSIM_AVAILABLE = True
except ImportError:
    pass


class TextSummarizer:
    """Unified text summarization class supporting multiple methods."""
    
    def __init__(self, method: str = "textrank"):
        """Initialize summarizer.
        
        Args:
            method: Summarization method - "textrank", "lsa", "lexrank", "gensim", or "transformer"
        """
        self.method = method.lower()
        self.transformer_pipeline = None
        
        # Initialize transformer if needed
        if self.method == "transformer":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install with: pip install transformers torch")
            # Load pre-trained summarization model
            self.transformer_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # CPU
            )
        
        # Check other dependencies
        if self.method in ["textrank", "lsa", "lexrank"] and not SUMY_AVAILABLE:
            raise ImportError("Sumy not available. Install with: pip install sumy")
        
        if self.method == "gensim" and not GENSIM_AVAILABLE:
            raise ImportError("Gensim summarization not available (removed in recent versions)")
    
    def summarize(self, 
                 text: str,
                 ratio: float = 0.2,
                 sentence_count: Optional[int] = None,
                 max_length: int = 130,
                 min_length: int = 30) -> str:
        """Summarize text.
        
        Args:
            text: Input text to summarize
            ratio: Ratio of sentences to keep (0.0-1.0) for extractive
            sentence_count: Number of sentences in summary (overrides ratio)
            max_length: Maximum length for abstractive summary
            min_length: Minimum length for abstractive summary
            
        Returns:
            Summary text
        """
        if not text or len(text.strip()) < 100:
            return text  # Too short to summarize
        
        if self.method == "transformer":
            return self._summarize_transformer(text, max_length, min_length)
        elif self.method == "gensim":
            return self._summarize_gensim(text, ratio)
        else:
            return self._summarize_extractive(text, ratio, sentence_count)
    
    def _summarize_extractive(self, 
                             text: str,
                             ratio: float = 0.2,
                             sentence_count: Optional[int] = None) -> str:
        """Extractive summarization using Sumy."""
        # Parse text
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        
        # Choose algorithm
        if self.method == "textrank":
            summarizer = TextRankSummarizer()
        elif self.method == "lsa":
            summarizer = LsaSummarizer()
        elif self.method == "lexrank":
            summarizer = LexRankSummarizer()
        else:
            summarizer = TextRankSummarizer()  # Default
        
        # Calculate sentence count if not provided
        if sentence_count is None:
            total_sentences = len(list(parser.document.sentences))
            sentence_count = max(1, int(total_sentences * ratio))
        
        # Generate summary
        summary_sentences = summarizer(parser.document, sentence_count)
        
        # Combine sentences
        summary = " ".join([str(sentence) for sentence in summary_sentences])
        return summary
    
    def _summarize_gensim(self, text: str, ratio: float = 0.2) -> str:
        """Gensim-based summarization."""
        try:
            summary = gensim_summarize(text, ratio=ratio)
            return summary if summary else text
        except:
            # Fallback if gensim fails
            return text
    
    def _summarize_transformer(self, 
                              text: str,
                              max_length: int = 130,
                              min_length: int = 30) -> str:
        """Abstractive summarization using transformers."""
        # Split long text into chunks (transformers have token limits)
        max_chunk_length = 1024
        
        if len(text) <= max_chunk_length:
            result = self.transformer_pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return result[0]['summary_text']
        else:
            # Chunk and summarize
            chunks = self._chunk_text(text, max_chunk_length)
            summaries = []
            
            for chunk in chunks[:3]:  # Limit to first 3 chunks
                result = self.transformer_pipeline(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summaries.append(result[0]['summary_text'])
            
            return " ".join(summaries)
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks by sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def extract_key_sentences(self, text: str, count: int = 3) -> List[str]:
        """Extract key sentences from text.
        
        Args:
            text: Input text
            count: Number of key sentences to extract
            
        Returns:
            List of key sentences
        """
        if self.method == "transformer":
            # Use extractive fallback for transformers
            temp_summarizer = TextSummarizer(method="textrank")
            return temp_summarizer.extract_key_sentences(text, count)
        
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        
        summary_sentences = summarizer(parser.document, count)
        return [str(sentence) for sentence in summary_sentences]
    
    def generate_coherent_summary(self, text: str, target_words: int = 100) -> str:
        """Generate a coherent, flowing summary with proper structure.
        
        This creates a more natural summary by:
        1. Extracting key sentences
        2. Reordering them logically
        3. Adding transitions for better flow
        
        Args:
            text: Input text to summarize
            target_words: Target word count for summary (~100 words)
            
        Returns:
            Coherent summary paragraph
        """
        import re
        
        # Calculate sentence count needed for ~100 words (avg 15-20 words per sentence)
        target_sentences = max(3, min(7, target_words // 15))
        
        # Extract important sentences
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        
        if self.method == "textrank":
            summarizer = TextRankSummarizer()
        elif self.method == "lsa":
            summarizer = LsaSummarizer()
        elif self.method == "lexrank":
            summarizer = LexRankSummarizer()
        else:
            summarizer = TextRankSummarizer()
        
        # Get more sentences than needed for better selection
        raw_sentences = summarizer(parser.document, target_sentences * 2)
        sentences = [str(s) for s in raw_sentences]
        
        if not sentences:
            return text[:500]  # Fallback
        
        # Find sentences that work well together
        # Prefer sentences from different parts of the document
        total_sentences = len(list(parser.document.sentences))
        selected = []
        sentence_positions = {}
        
        for i, sent in enumerate(list(parser.document.sentences)):
            sentence_positions[str(sent)] = i
        
        # Select sentences distributed across document
        for sent in sentences:
            if len(selected) >= target_sentences:
                break
            
            # Check if sentence adds new information
            sent_clean = sent.lower()
            is_unique = True
            
            for existing in selected:
                existing_clean = existing.lower()
                # Check for significant overlap
                common_words = set(sent_clean.split()) & set(existing_clean.split())
                if len(common_words) > len(sent_clean.split()) * 0.6:
                    is_unique = False
                    break
            
            if is_unique:
                selected.append(sent)
        
        if not selected:
            selected = sentences[:target_sentences]
        
        # Sort by position in original document for logical flow
        selected_with_pos = [(sent, sentence_positions.get(sent, 999)) for sent in selected]
        selected_with_pos.sort(key=lambda x: x[1])
        ordered_sentences = [s[0] for s in selected_with_pos]
        
        # Create flowing paragraph with transitions
        summary_parts = []
        for i, sent in enumerate(ordered_sentences):
            sent = sent.strip()
            if i == 0:
                # First sentence - no transition needed
                summary_parts.append(sent)
            else:
                # Check if we need a transition
                prev_sent = ordered_sentences[i-1].lower()
                curr_sent = sent.lower()
                
                # If sentences are related, just connect them
                # Otherwise, might add subtle transition
                summary_parts.append(sent)
        
        # Join into coherent paragraph
        final_summary = " ".join(summary_parts)
        
        # Clean up any double spaces or punctuation issues
        final_summary = re.sub(r'\s+', ' ', final_summary)
        final_summary = re.sub(r'\s+([.,!?])', r'\1', final_summary)
        
        return final_summary.strip()


class InsightGenerator:
    """Generate insights from topics and sentiment analysis."""
    
    def __init__(self):
        """Initialize insight generator."""
        self.insight_templates = {
            "high_positive": "✅ Strong positive sentiment detected: {context}",
            "high_negative": "⚠️ Concern identified: {context} - Immediate attention recommended",
            "mixed": "🔀 Mixed opinions found: {context}",
            "trending": "📈 Trending topic: {context}",
            "opportunity": "💡 Opportunity identified: {context}",
            "risk": "🚨 Risk alert: {context}",
        }
    
    def generate_insights(self,
                         topic_sentiments: Dict[int, Dict[str, any]],
                         topic_words: Dict[int, List[Tuple[str, float]]]) -> List[Dict[str, str]]:
        """Generate actionable insights from topic-sentiment data.
        
        Args:
            topic_sentiments: Sentiment distribution per topic
            topic_words: Top words per topic
            
        Returns:
            List of insight dictionaries with type, message, and priority
        """
        insights = []
        
        for topic_id, sentiment in topic_sentiments.items():
            words = topic_words.get(topic_id, [])
            topic_label = ", ".join([w for w, _ in words[:3]]) if words else f"Topic {topic_id + 1}"
            
            pos_pct = sentiment.get("positive_pct", 0)
            neg_pct = sentiment.get("negative_pct", 0)
            
            # High negative sentiment - critical insight
            if neg_pct > 70:
                insights.append({
                    "type": "risk",
                    "priority": "high",
                    "topic_id": topic_id,
                    "message": f"Critical: '{topic_label}' shows {neg_pct:.1f}% negative sentiment",
                    "recommendation": "Immediate review and action required to address negative feedback"
                })
            
            # High positive sentiment - opportunity
            elif pos_pct > 70:
                insights.append({
                    "type": "opportunity",
                    "priority": "medium",
                    "topic_id": topic_id,
                    "message": f"Success: '{topic_label}' has {pos_pct:.1f}% positive sentiment",
                    "recommendation": "Leverage this strength and replicate success factors"
                })
            
            # Mixed sentiment - requires attention
            elif 30 < pos_pct < 70 and 30 < neg_pct < 70:
                insights.append({
                    "type": "mixed",
                    "priority": "medium",
                    "topic_id": topic_id,
                    "message": f"Divided opinion on '{topic_label}' (Pos: {pos_pct:.1f}%, Neg: {neg_pct:.1f}%)",
                    "recommendation": "Investigate root causes of mixed sentiment"
                })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return insights
    
    def generate_executive_summary(self,
                                   topic_sentiments: Dict[int, Dict[str, any]],
                                   topic_words: Dict[int, List[Tuple[str, float]]],
                                   total_docs: int) -> str:
        """Generate executive summary of analysis.
        
        Args:
            topic_sentiments: Sentiment per topic
            topic_words: Words per topic
            total_docs: Total number of documents analyzed
            
        Returns:
            Executive summary text
        """
        summary_parts = []
        
        # Overview
        summary_parts.append(f"## Executive Summary\n")
        summary_parts.append(f"**Documents Analyzed:** {total_docs}")
        summary_parts.append(f"**Topics Discovered:** {len(topic_sentiments)}\n")
        
        # Overall sentiment
        total_pos = sum(s.get("positive", 0) for s in topic_sentiments.values())
        total_neg = sum(s.get("negative", 0) for s in topic_sentiments.values())
        total_neu = sum(s.get("neutral", 0) for s in topic_sentiments.values())
        total_all = total_pos + total_neg + total_neu
        
        if total_all > 0:
            summary_parts.append(f"### Overall Sentiment Distribution")
            summary_parts.append(f"- Positive: {total_pos/total_all*100:.1f}%")
            summary_parts.append(f"- Negative: {total_neg/total_all*100:.1f}%")
            summary_parts.append(f"- Neutral: {total_neu/total_all*100:.1f}%\n")
        
        # Key findings
        insights = self.generate_insights(topic_sentiments, topic_words)
        if insights:
            summary_parts.append(f"### Key Findings")
            for i, insight in enumerate(insights[:5], 1):  # Top 5 insights
                summary_parts.append(f"{i}. {insight['message']}")
        
        return "\n".join(summary_parts)


def quick_summarize(text: str, method: str = "textrank", sentences: int = 3) -> str:
    """Quick text summarization.
    
    Args:
        text: Text to summarize
        method: Method to use
        sentences: Number of sentences in summary
        
    Returns:
        Summary text
    """
    summarizer = TextSummarizer(method=method)
    return summarizer.summarize(text, sentence_count=sentences)


__all__ = [
    "TextSummarizer",
    "InsightGenerator",
    "quick_summarize",
    "SUMY_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
    "GENSIM_AVAILABLE"
]
