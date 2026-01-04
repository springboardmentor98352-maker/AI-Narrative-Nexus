"""
Sentiment Analysis Module for NarrativeNexus

This module provides sentiment analysis capabilities using multiple approaches:
- VADER (rule-based, great for social media)
- TextBlob (simple, general-purpose)
- Transformers (BERT-based, most accurate)

Sentiment Analysis vs TextBlob vs VADER vs Transformers:
--------------------------------------------------------
VADER (Valence Aware Dictionary and sEntiment Reasoner):
- Rule-based lexicon approach
- Excellent for social media text (handles emojis, slang, capitalization)
- Fast and lightweight
- Good for real-time applications
- Compound score: -1 (most negative) to +1 (most positive)

TextBlob:
- Simple pattern-based approach
- Good for general text
- Fast and easy to use
- Returns polarity (-1 to +1) and subjectivity (0 to 1)
- Less accurate than transformers but much faster

Transformers (BERT-based):
- Deep learning models trained on large datasets
- Most accurate for complex sentiment
- Handles context and nuance better
- Slower and requires more computational resources
- Best for production-quality results

Recommendation: Use VADER for quick analysis, Transformers for accuracy
"""

from typing import Dict, List, Tuple, Optional
from collections import Counter
import os

# Try importing different sentiment analysis libraries
VADER_AVAILABLE = False
TEXTBLOB_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    pass

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class SentimentAnalyzer:
    """Unified sentiment analysis class supporting multiple methods."""
    
    def __init__(self, method: str = "vader"):
        """Initialize sentiment analyzer.
        
        Args:
            method: Analysis method - "vader", "textblob", or "transformer"
        """
        self.method = method.lower()
        self.vader_analyzer = None
        self.transformer_pipeline = None
        
        # Initialize chosen method
        if self.method == "vader":
            if not VADER_AVAILABLE:
                raise ImportError("VADER not available. Install with: pip install vaderSentiment")
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        elif self.method == "textblob":
            if not TEXTBLOB_AVAILABLE:
                raise ImportError("TextBlob not available. Install with: pip install textblob")
        
        elif self.method == "transformer":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install with: pip install transformers torch")
            # Load pre-trained sentiment model
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU
            )
    
    def analyze(self, text: str) -> Dict[str, any]:
        """Analyze sentiment of text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment label, score, and confidence
        """
        if not text or len(text.strip()) == 0:
            return {
                "label": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "method": self.method
            }
        
        if self.method == "vader":
            return self._analyze_vader(text)
        elif self.method == "textblob":
            return self._analyze_textblob(text)
        elif self.method == "transformer":
            return self._analyze_transformer(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _analyze_vader(self, text: str) -> Dict[str, any]:
        """Analyze using VADER."""
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Classify based on compound score
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "label": label,
            "score": compound,
            "confidence": abs(compound),
            "positive": scores['pos'],
            "negative": scores['neg'],
            "neutral": scores['neu'],
            "method": "vader"
        }
    
    def _analyze_textblob(self, text: str) -> Dict[str, any]:
        """Analyze using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify based on polarity
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "label": label,
            "score": polarity,
            "confidence": abs(polarity),
            "subjectivity": subjectivity,
            "method": "textblob"
        }
    
    def _analyze_transformer(self, text: str) -> Dict[str, any]:
        """Analyze using transformer model."""
        # Limit text length for transformer (max 512 tokens)
        if len(text) > 2000:
            text = text[:2000]
        
        result = self.transformer_pipeline(text)[0]
        label = result['label'].lower()
        score = result['score']
        
        # Normalize to positive/negative/neutral
        if label == "positive":
            sentiment_score = score
        else:  # negative
            sentiment_score = -score
        
        # Add neutral category for low confidence
        if score < 0.6:
            label = "neutral"
            sentiment_score = 0.0
        
        return {
            "label": label,
            "score": sentiment_score,
            "confidence": score,
            "method": "transformer"
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """Analyze multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment dictionaries
        """
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results
    
    def get_sentiment_distribution(self, texts: List[str]) -> Dict[str, any]:
        """Get overall sentiment distribution for multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with distribution statistics
        """
        sentiments = self.analyze_batch(texts)
        
        # Count labels
        label_counts = Counter([s['label'] for s in sentiments])
        total = len(sentiments)
        
        # Calculate average scores
        avg_score = sum([s['score'] for s in sentiments]) / total if total > 0 else 0
        avg_confidence = sum([s['confidence'] for s in sentiments]) / total if total > 0 else 0
        
        return {
            "total_texts": total,
            "positive": label_counts.get("positive", 0),
            "negative": label_counts.get("negative", 0),
            "neutral": label_counts.get("neutral", 0),
            "positive_pct": (label_counts.get("positive", 0) / total * 100) if total > 0 else 0,
            "negative_pct": (label_counts.get("negative", 0) / total * 100) if total > 0 else 0,
            "neutral_pct": (label_counts.get("neutral", 0) / total * 100) if total > 0 else 0,
            "avg_score": avg_score,
            "avg_confidence": avg_confidence,
            "method": self.method
        }


class TopicSentimentAnalyzer:
    """Analyze sentiment for topics discovered by topic modeling."""
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        """Initialize with a sentiment analyzer.
        
        Args:
            sentiment_analyzer: SentimentAnalyzer instance
        """
        self.sentiment_analyzer = sentiment_analyzer
    
    def analyze_topic_sentiment(self, 
                               topic_documents: Dict[int, List[str]]) -> Dict[int, Dict[str, any]]:
        """Analyze sentiment for each topic's documents.
        
        Args:
            topic_documents: Dictionary mapping topic_id to list of document texts
            
        Returns:
            Dictionary mapping topic_id to sentiment distribution
        """
        topic_sentiments = {}
        
        for topic_id, documents in topic_documents.items():
            if not documents:
                continue
            
            distribution = self.sentiment_analyzer.get_sentiment_distribution(documents)
            topic_sentiments[topic_id] = distribution
        
        return topic_sentiments
    
    def get_dominant_sentiment_per_topic(self,
                                         topic_documents: Dict[int, List[str]]) -> Dict[int, str]:
        """Get the dominant sentiment for each topic.
        
        Args:
            topic_documents: Dictionary mapping topic_id to list of document texts
            
        Returns:
            Dictionary mapping topic_id to dominant sentiment label
        """
        topic_sentiments = self.analyze_topic_sentiment(topic_documents)
        dominant = {}
        
        for topic_id, sentiment_dist in topic_sentiments.items():
            # Find label with highest count
            counts = {
                "positive": sentiment_dist.get("positive", 0),
                "negative": sentiment_dist.get("negative", 0),
                "neutral": sentiment_dist.get("neutral", 0)
            }
            dominant[topic_id] = max(counts, key=counts.get)
        
        return dominant
    
    def generate_sentiment_insights(self,
                                    topic_sentiments: Dict[int, Dict[str, any]],
                                    topic_words: Dict[int, List[Tuple[str, float]]]) -> List[str]:
        """Generate human-readable insights from topic-sentiment analysis.
        
        Args:
            topic_sentiments: Dictionary from analyze_topic_sentiment()
            topic_words: Dictionary mapping topic_id to list of (word, score) tuples
            
        Returns:
            List of insight strings
        """
        insights = []
        
        for topic_id, sentiment in topic_sentiments.items():
            # Get topic words for context
            words = topic_words.get(topic_id, [])
            if words:
                top_words = ", ".join([w for w, _ in words[:3]])
            else:
                top_words = f"Topic {topic_id + 1}"
            
            # Determine dominant sentiment
            pos_pct = sentiment.get("positive_pct", 0)
            neg_pct = sentiment.get("negative_pct", 0)
            neu_pct = sentiment.get("neutral_pct", 0)
            
            if pos_pct > 60:
                insight = f"✅ **Topic {topic_id + 1}** ({top_words}): **Predominantly Positive** ({pos_pct:.1f}% positive sentiment)"
            elif neg_pct > 60:
                insight = f"⚠️ **Topic {topic_id + 1}** ({top_words}): **Predominantly Negative** ({neg_pct:.1f}% negative sentiment) - Requires attention"
            elif neu_pct > 60:
                insight = f"ℹ️ **Topic {topic_id + 1}** ({top_words}): **Mostly Neutral** ({neu_pct:.1f}% neutral)"
            else:
                insight = f"🔀 **Topic {topic_id + 1}** ({top_words}): **Mixed Sentiments** (Pos: {pos_pct:.1f}%, Neg: {neg_pct:.1f}%, Neu: {neu_pct:.1f}%)"
            
            insights.append(insight)
        
        return insights


def quick_sentiment_analysis(text: str, method: str = "vader") -> str:
    """Quick sentiment analysis for single text.
    
    Args:
        text: Text to analyze
        method: Method to use ("vader", "textblob", or "transformer")
        
    Returns:
        Sentiment label (positive/negative/neutral)
    """
    analyzer = SentimentAnalyzer(method=method)
    result = analyzer.analyze(text)
    return result['label']


def batch_sentiment_analysis(texts: List[str], 
                            method: str = "vader") -> List[Dict[str, any]]:
    """Batch sentiment analysis.
    
    Args:
        texts: List of texts to analyze
        method: Method to use
        
    Returns:
        List of sentiment dictionaries
    """
    analyzer = SentimentAnalyzer(method=method)
    return analyzer.analyze_batch(texts)


__all__ = [
    "SentimentAnalyzer",
    "TopicSentimentAnalyzer",
    "quick_sentiment_analysis",
    "batch_sentiment_analysis",
    "VADER_AVAILABLE",
    "TEXTBLOB_AVAILABLE",
    "TRANSFORMERS_AVAILABLE"
]
