"""
Unit tests for Sentiment Analysis module
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from narrativenexus_sentiment import (
    SentimentAnalyzer, TopicSentimentAnalyzer,
    VADER_AVAILABLE, TEXTBLOB_AVAILABLE
)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is absolutely amazing! I love it!",
        "Terrible experience. Very disappointed.",
        "It's okay, nothing special.",
        "Fantastic product! Highly recommended!",
        "Worst purchase ever. Complete waste of money."
    ]


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer class."""
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_vader_initialization(self):
        """Test VADER analyzer initialization."""
        analyzer = SentimentAnalyzer(method="vader")
        assert analyzer.method == "vader"
        assert analyzer.vader_analyzer is not None
    
    @pytest.mark.skipif(not TEXTBLOB_AVAILABLE, reason="TextBlob not installed")
    def test_textblob_initialization(self):
        """Test TextBlob analyzer initialization."""
        analyzer = SentimentAnalyzer(method="textblob")
        assert analyzer.method == "textblob"
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_analyze_positive(self):
        """Test analyzing positive sentiment."""
        analyzer = SentimentAnalyzer(method="vader")
        result = analyzer.analyze("This is amazing! I love it!")
        
        assert result['label'] == 'positive'
        assert result['score'] > 0
        assert 'confidence' in result
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_analyze_negative(self):
        """Test analyzing negative sentiment."""
        analyzer = SentimentAnalyzer(method="vader")
        result = analyzer.analyze("This is terrible! I hate it!")
        
        assert result['label'] == 'negative'
        assert result['score'] < 0
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_analyze_neutral(self):
        """Test analyzing neutral sentiment."""
        analyzer = SentimentAnalyzer(method="vader")
        result = analyzer.analyze("This is a thing.")
        
        assert result['label'] == 'neutral'
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_analyze_empty(self):
        """Test analyzing empty text."""
        analyzer = SentimentAnalyzer(method="vader")
        result = analyzer.analyze("")
        
        assert result['label'] == 'neutral'
        assert result['score'] == 0.0
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_analyze_batch(self, sample_texts):
        """Test batch analysis."""
        analyzer = SentimentAnalyzer(method="vader")
        results = analyzer.analyze_batch(sample_texts)
        
        assert len(results) == len(sample_texts)
        assert all('label' in r for r in results)
        assert all('score' in r for r in results)
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_get_distribution(self, sample_texts):
        """Test sentiment distribution calculation."""
        analyzer = SentimentAnalyzer(method="vader")
        distribution = analyzer.get_sentiment_distribution(sample_texts)
        
        assert 'total_texts' in distribution
        assert distribution['total_texts'] == len(sample_texts)
        assert 'positive' in distribution
        assert 'negative' in distribution
        assert 'neutral' in distribution
        assert 'positive_pct' in distribution
        assert 'avg_score' in distribution


class TestTopicSentimentAnalyzer:
    """Test cases for TopicSentimentAnalyzer class."""
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_initialization(self):
        """Test initialization."""
        sentiment_analyzer = SentimentAnalyzer(method="vader")
        topic_analyzer = TopicSentimentAnalyzer(sentiment_analyzer)
        
        assert topic_analyzer.sentiment_analyzer is not None
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_analyze_topic_sentiment(self):
        """Test analyzing sentiment per topic."""
        sentiment_analyzer = SentimentAnalyzer(method="vader")
        topic_analyzer = TopicSentimentAnalyzer(sentiment_analyzer)
        
        topic_docs = {
            0: ["This is great!", "Amazing product!"],
            1: ["Terrible quality.", "Very disappointed."]
        }
        
        results = topic_analyzer.analyze_topic_sentiment(topic_docs)
        
        assert len(results) == 2
        assert 0 in results
        assert 1 in results
        assert results[0]['positive_pct'] > results[1]['positive_pct']
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_get_dominant_sentiment(self):
        """Test getting dominant sentiment per topic."""
        sentiment_analyzer = SentimentAnalyzer(method="vader")
        topic_analyzer = TopicSentimentAnalyzer(sentiment_analyzer)
        
        topic_docs = {
            0: ["Excellent!", "Great!", "Love it!"],
            1: ["Bad.", "Terrible.", "Awful."]
        }
        
        dominant = topic_analyzer.get_dominant_sentiment_per_topic(topic_docs)
        
        assert dominant[0] == "positive"
        assert dominant[1] == "negative"
    
    @pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not installed")
    def test_generate_insights(self):
        """Test insight generation."""
        sentiment_analyzer = SentimentAnalyzer(method="vader")
        topic_analyzer = TopicSentimentAnalyzer(sentiment_analyzer)
        
        topic_sentiments = {
            0: {"positive_pct": 80, "negative_pct": 10, "neutral_pct": 10}
        }
        topic_words = {
            0: [("great", 0.5), ("excellent", 0.4)]
        }
        
        insights = topic_analyzer.generate_sentiment_insights(topic_sentiments, topic_words)
        
        assert len(insights) > 0
        assert all(isinstance(i, str) for i in insights)
