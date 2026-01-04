"""
Unit tests for Topic Modeling module
"""

import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from narrativenexus_topic_modeling import TopicModelManager, GENSIM_AVAILABLE, SKLEARN_AVAILABLE


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret visual information",
        "Reinforcement learning involves training agents through rewards"
    ]


@pytest.fixture
def tokenized_docs():
    """Sample tokenized documents."""
    return [
        ["machine", "learning", "subset", "artificial", "intelligence"],
        ["deep", "learning", "neural", "network", "multiple", "layer"],
        ["natural", "language", "processing", "computer", "understand", "human", "language"],
        ["computer", "vision", "machine", "interpret", "visual", "information"],
        ["reinforcement", "learning", "training", "agent", "reward"]
    ]


class TestTopicModelManager:
    """Test cases for TopicModelManager class."""
    
    def test_initialization(self):
        """Test topic model manager initialization."""
        tm = TopicModelManager(data_dir="test_models")
        assert tm.data_dir == "test_models"
        assert tm.dictionary is None
        assert tm.corpus is None
    
    @pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not installed")
    def test_create_dictionary(self, tokenized_docs):
        """Test dictionary creation."""
        tm = TopicModelManager()
        tm.create_dictionary(tokenized_docs, no_below=1, no_above=1.0)
        
        assert tm.dictionary is not None
        assert len(tm.dictionary) > 0
    
    @pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not installed")
    def test_create_corpus(self, tokenized_docs):
        """Test corpus creation."""
        tm = TopicModelManager()
        tm.create_dictionary(tokenized_docs, no_below=1, no_above=1.0)
        tm.create_corpus(tokenized_docs)
        
        assert tm.corpus is not None
        assert len(list(tm.corpus)) == len(tokenized_docs)
    
    @pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not installed")
    def test_lda_training(self, tokenized_docs):
        """Test LDA model training."""
        tm = TopicModelManager()
        tm.create_dictionary(tokenized_docs, no_below=1, no_above=1.0)
        tm.create_corpus(tokenized_docs)
        
        model = tm.train_lda_model(num_topics=2, passes=5, iterations=50)
        
        assert model is not None
        assert tm.lda_model is not None
        assert tm.lda_model.num_topics == 2
    
    @pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not installed")
    def test_get_lda_topics(self, tokenized_docs):
        """Test getting topics from LDA model."""
        tm = TopicModelManager()
        tm.create_dictionary(tokenized_docs, no_below=1, no_above=1.0)
        tm.create_corpus(tokenized_docs)
        tm.train_lda_model(num_topics=2, passes=5, iterations=50)
        
        topics = tm.get_lda_topics(num_words=5)
        
        assert len(topics) == 2
        assert all(isinstance(t, tuple) for t in topics)
        assert all(len(t[1]) == 5 for t in topics)
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not installed")
    def test_nmf_training(self, sample_documents):
        """Test NMF model training."""
        tm = TopicModelManager()
        
        model = tm.train_nmf_model(sample_documents, num_topics=2, max_features=50)
        
        assert model is not None
        assert tm.nmf_model is not None
        assert tm.nmf_model.n_components == 2
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not installed")
    def test_get_nmf_topics(self, sample_documents):
        """Test getting topics from NMF model."""
        tm = TopicModelManager()
        tm.train_nmf_model(sample_documents, num_topics=2, max_features=50)
        
        topics = tm.get_nmf_topics(num_words=5)
        
        assert len(topics) == 2
        assert all(isinstance(t, tuple) for t in topics)
    
    def test_stats(self, tokenized_docs):
        """Test getting statistics."""
        tm = TopicModelManager()
        
        if GENSIM_AVAILABLE:
            tm.create_dictionary(tokenized_docs, no_below=1, no_above=1.0)
            tm.create_corpus(tokenized_docs)
            
            stats = tm.get_stats()
            
            assert 'vocab_size' in stats
            assert stats['vocab_size'] > 0
