"""
Topic Modeling Module for NarrativeNexus

This module provides implementations for:
- LDA (Latent Dirichlet Allocation) using Gensim
- NMF (Non-negative Matrix Factorization) using sklearn
- Dictionary and corpus management
- Topic visualization and analysis

LDA vs NMF Comparison:
-------------------
LDA (Latent Dirichlet Allocation):
- Probabilistic model based on Bayesian statistics
- Assumes documents are mixtures of topics, topics are mixtures of words
- Better for interpretable topics with overlapping themes
- Works well with smaller datasets
- Outputs probability distributions
- Primary choice for most text analysis tasks

NMF (Non-negative Matrix Factorization):
- Linear algebra-based matrix factorization
- Faster computation than LDA
- Produces sparser, more distinct topics
- Better for large-scale datasets
- Topics tend to be more separated/exclusive
- Good backup when LDA is too slow or results are unclear
"""

import os
import pickle
from typing import List, Tuple, Dict, Optional
from collections import Counter

# LDA - Primary Algorithm (using Gensim)
try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel, CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# NMF - Backup Algorithm (using sklearn)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF as SklearnNMF
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TopicModelManager:
    """Manages dictionary, corpus, and topic models for text analysis."""
    
    def __init__(self, data_dir: str = "topic_models"):
        """Initialize the topic model manager.
        
        Args:
            data_dir: Directory to save/load models and data
        """
        self.data_dir = data_dir
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.nmf_model = None
        self.vectorizer = None
        
        # Ensure directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def create_dictionary(self, tokenized_docs: List[List[str]], 
                         no_below: int = 2, 
                         no_above: float = 0.5,
                         keep_n: int = 100000) -> None:
        """Create Gensim dictionary from tokenized documents.
        
        Args:
            tokenized_docs: List of tokenized documents (list of token lists)
            no_below: Keep tokens appearing in at least this many documents
            no_above: Keep tokens appearing in no more than this fraction of documents
            keep_n: Keep only the top N most frequent tokens
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is not installed. Install with: pip install gensim")
        
        # Create dictionary (word-to-ID mapping)
        self.dictionary = corpora.Dictionary(tokenized_docs)
        
        # Filter extremes
        self.dictionary.filter_extremes(
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n
        )
        
        # Compactify dictionary to remove gaps in IDs
        self.dictionary.compactify()
    
    def create_corpus(self, tokenized_docs: List[List[str]]) -> None:
        """Create bag-of-words corpus from tokenized documents.
        
        Args:
            tokenized_docs: List of tokenized documents
        """
        if self.dictionary is None:
            raise ValueError("Dictionary must be created first. Call create_dictionary()")
        
        # Convert documents to bag-of-words representation
        # Each doc becomes list of (token_id, token_count) tuples
        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
    
    def save_dictionary_and_corpus(self, prefix: str = "model") -> Tuple[str, str]:
        """Save dictionary and corpus to disk.
        
        Args:
            prefix: Filename prefix for saved files
            
        Returns:
            Tuple of (dictionary_path, corpus_path)
        """
        if self.dictionary is None or self.corpus is None:
            raise ValueError("Dictionary and corpus must be created before saving")
        
        dict_path = os.path.join(self.data_dir, f"{prefix}_dictionary.dict")
        corpus_path = os.path.join(self.data_dir, f"{prefix}_corpus.mm")
        
        # Save dictionary
        self.dictionary.save(dict_path)
        
        # Save corpus in Matrix Market format (efficient for sparse matrices)
        corpora.MmCorpus.serialize(corpus_path, self.corpus)
        
        return dict_path, corpus_path
    
    def load_dictionary_and_corpus(self, prefix: str = "model") -> None:
        """Load dictionary and corpus from disk.
        
        Args:
            prefix: Filename prefix of saved files
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is not installed")
        
        dict_path = os.path.join(self.data_dir, f"{prefix}_dictionary.dict")
        corpus_path = os.path.join(self.data_dir, f"{prefix}_corpus.mm")
        
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        # Load dictionary
        self.dictionary = corpora.Dictionary.load(dict_path)
        
        # Load corpus
        self.corpus = corpora.MmCorpus(corpus_path)
    
    def train_lda_model(self, 
                       num_topics: int = 10,
                       passes: int = 15,
                       iterations: int = 400,
                       alpha: str = 'auto',
                       eta: str = 'auto',
                       random_state: int = 42) -> LdaModel:
        """Train LDA model on the corpus.
        
        Args:
            num_topics: Number of topics to extract
            passes: Number of passes through the corpus during training
            iterations: Maximum number of iterations through the corpus
            alpha: Document-topic density (auto = learn from data)
            eta: Topic-word density (auto = learn from data)
            random_state: Random seed for reproducibility
            
        Returns:
            Trained LDA model
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is not installed")
        
        if self.corpus is None or self.dictionary is None:
            raise ValueError("Corpus and dictionary must be created first")
        
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            alpha=alpha,
            eta=eta,
            random_state=random_state,
            per_word_topics=True,
            chunksize=2000,  # Process in larger chunks for efficiency
            eval_every=None  # Don't evaluate perplexity during training (faster)
        )
        
        return self.lda_model
    
    def train_nmf_model(self,
                       documents: List[str],
                       num_topics: int = 10,
                       max_features: int = 1000,
                       random_state: int = 42) -> SklearnNMF:
        """Train NMF model on documents.
        
        Args:
            documents: List of text documents (not tokenized)
            num_topics: Number of topics to extract
            max_features: Maximum number of features (words) to consider
            random_state: Random seed for reproducibility
            
        Returns:
            Trained NMF model
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is not installed")
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Train NMF model
        self.nmf_model = SklearnNMF(
            n_components=num_topics,
            random_state=random_state,
            max_iter=400
        )
        
        self.nmf_model.fit(tfidf_matrix)
        
        return self.nmf_model
    
    def get_lda_topics(self, num_words: int = 10) -> List[Tuple[int, List[Tuple[str, float]]]]:
        """Get topics from trained LDA model.
        
        Args:
            num_words: Number of top words per topic
            
        Returns:
            List of (topic_id, [(word, probability), ...])
        """
        if self.lda_model is None:
            raise ValueError("LDA model must be trained first")
        
        topics = []
        for topic_id in range(self.lda_model.num_topics):
            topic_words = self.lda_model.show_topic(topic_id, topn=num_words)
            topics.append((topic_id, topic_words))
        
        return topics
    
    def get_nmf_topics(self, num_words: int = 10) -> List[Tuple[int, List[Tuple[str, float]]]]:
        """Get topics from trained NMF model.
        
        Args:
            num_words: Number of top words per topic
            
        Returns:
            List of (topic_id, [(word, weight), ...])
        """
        if self.nmf_model is None or self.vectorizer is None:
            raise ValueError("NMF model must be trained first")
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_id, topic in enumerate(self.nmf_model.components_):
            top_indices = topic.argsort()[-num_words:][::-1]
            top_words = [(feature_names[i], topic[i]) for i in top_indices]
            topics.append((topic_id, top_words))
        
        return topics
    
    def get_document_topics_lda(self, doc_bow) -> List[Tuple[int, float]]:
        """Get topic distribution for a document using LDA.
        
        Args:
            doc_bow: Document in bag-of-words format
            
        Returns:
            List of (topic_id, probability)
        """
        if self.lda_model is None:
            raise ValueError("LDA model must be trained first")
        
        return self.lda_model.get_document_topics(doc_bow)
    
    def compute_coherence_score(self, tokenized_docs: List[List[str]]) -> float:
        """Compute coherence score for LDA model.
        
        Higher coherence scores indicate better topic quality.
        
        Args:
            tokenized_docs: Original tokenized documents
            
        Returns:
            Coherence score (C_v metric)
        """
        if self.lda_model is None:
            raise ValueError("LDA model must be trained first")
        
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is not installed")
        
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=tokenized_docs,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        
        return coherence_model.get_coherence()
    
    def compute_perplexity(self) -> float:
        """Compute perplexity score for LDA model.
        
        Lower perplexity indicates better model performance.
        
        Returns:
            Perplexity score
        """
        if self.lda_model is None:
            raise ValueError("LDA model must be trained first")
        
        if self.corpus is None:
            raise ValueError("Corpus must be created first")
        
        return self.lda_model.log_perplexity(self.corpus)
    
    def compute_nmf_coherence(self, tokenized_docs: List[List[str]]) -> float:
        """Compute coherence score for NMF model.
        
        Uses the same C_v coherence metric as LDA for fair comparison.
        
        Args:
            tokenized_docs: Original tokenized documents
            
        Returns:
            Coherence score (C_v metric)
        """
        if self.nmf_model is None:
            raise ValueError("NMF model must be trained first")
        
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is not installed for coherence computation")
        
        # Get top words for each NMF topic
        nmf_topics = self.get_nmf_topics(num_words=20)
        
        # Convert to format expected by CoherenceModel
        topics_words = [[word for word, _ in topic_words] for _, topic_words in nmf_topics]
        
        # Create a temporary dictionary from tokenized docs
        temp_dict = corpora.Dictionary(tokenized_docs)
        
        # Compute coherence using gensim's CoherenceModel
        coherence_model = CoherenceModel(
            topics=topics_words,
            texts=tokenized_docs,
            dictionary=temp_dict,
            coherence='c_v'
        )
        
        return coherence_model.get_coherence()
    
    def get_topic_distribution_matrix(self) -> List[List[float]]:
        """Get topic distribution for all documents in corpus.
        
        Returns:
            List of topic distributions (one per document)
        """
        if self.lda_model is None:
            raise ValueError("LDA model must be trained first")
        
        if self.corpus is None:
            raise ValueError("Corpus must be created first")
        
        doc_topics = []
        num_topics = self.lda_model.num_topics
        
        for doc in self.corpus:
            topic_dist = [0.0] * num_topics
            for topic_id, prob in self.lda_model.get_document_topics(doc):
                topic_dist[topic_id] = prob
            doc_topics.append(topic_dist)
        
        return doc_topics
    
    def save_lda_model(self, filename: str = "lda_model") -> str:
        """Save trained LDA model to disk.
        
        Args:
            filename: Name of the model file
            
        Returns:
            Path to saved model
        """
        if self.lda_model is None:
            raise ValueError("LDA model must be trained first")
        
        model_path = os.path.join(self.data_dir, f"{filename}.model")
        self.lda_model.save(model_path)
        
        return model_path
    
    def load_lda_model(self, filename: str = "lda_model") -> None:
        """Load LDA model from disk.
        
        Args:
            filename: Name of the model file
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is not installed")
        
        model_path = os.path.join(self.data_dir, f"{filename}.model")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.lda_model = LdaModel.load(model_path)
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the current dictionary and corpus.
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        if self.dictionary:
            stats['vocab_size'] = len(self.dictionary)
            stats['num_documents'] = self.dictionary.num_docs if hasattr(self.dictionary, 'num_docs') else 0
        
        if self.corpus:
            stats['corpus_size'] = len(list(self.corpus)) if hasattr(self.corpus, '__len__') else 0
        
        if self.lda_model:
            stats['lda_num_topics'] = self.lda_model.num_topics
        
        if self.nmf_model:
            stats['nmf_num_topics'] = self.nmf_model.n_components
        
        return stats
    
    def create_topic_word_visualization(self, num_words: int = 10, use_lda: bool = True):
        """Create bar chart visualization for topic-word distributions.
        
        Args:
            num_words: Number of top words to show per topic
            use_lda: Use LDA model if True, NMF if False
            
        Returns:
            Matplotlib figure object
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        if use_lda:
            if self.lda_model is None:
                raise ValueError("LDA model must be trained first")
            topics = self.get_lda_topics(num_words=num_words)
        else:
            if self.nmf_model is None:
                raise ValueError("NMF model must be trained first")
            topics = self.get_nmf_topics(num_words=num_words)
        
        num_topics = len(topics)
        fig, axes = plt.subplots(num_topics, 1, figsize=(12, num_topics * 2))
        
        if num_topics == 1:
            axes = [axes]
        
        for idx, (topic_id, words) in enumerate(topics):
            word_labels = [w for w, _ in words]
            word_weights = [p for _, p in words]
            
            axes[idx].barh(word_labels, word_weights, color='#667eea')
            axes[idx].set_xlabel('Weight/Probability', fontsize=10)
            axes[idx].set_title(f'Topic {topic_id + 1}', fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def get_ensemble_topics(self, 
                           tokenized_docs: List[List[str]],
                           documents: List[str],
                           num_topics: int = 10,
                           num_words: int = 10) -> List[Tuple[int, List[Tuple[str, float]]]]:
        """Combine LDA and NMF topics using ensemble approach.
        
        Trains both models and combines their top words with averaged weights.
        
        Args:
            tokenized_docs: Tokenized documents for LDA
            documents: Raw documents for NMF
            num_topics: Number of topics
            num_words: Number of words per topic
            
        Returns:
            List of (topic_id, words) combining both models
        """
        # Train both models if not already trained
        if self.lda_model is None:
            self.create_dictionary(tokenized_docs)
            self.create_corpus(tokenized_docs)
            self.train_lda_model(num_topics=num_topics)
        
        if self.nmf_model is None:
            self.train_nmf_model(documents, num_topics=num_topics)
        
        # Get topics from both models
        lda_topics = self.get_lda_topics(num_words=num_words * 2)  # Get more words
        nmf_topics = self.get_nmf_topics(num_words=num_words * 2)
        
        # Combine topics
        ensemble_topics = []
        
        for topic_id in range(num_topics):
            # Merge word lists from both models
            word_scores = {}
            
            # Add LDA words
            if topic_id < len(lda_topics):
                for word, score in lda_topics[topic_id][1]:
                    word_scores[word] = word_scores.get(word, 0) + score * 0.5
            
            # Add NMF words
            if topic_id < len(nmf_topics):
                for word, score in nmf_topics[topic_id][1]:
                    # Normalize NMF scores to similar range as LDA
                    normalized_score = score / max([s for _, s in nmf_topics[topic_id][1]])
                    word_scores[word] = word_scores.get(word, 0) + normalized_score * 0.5
            
            # Sort by combined score
            top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:num_words]
            ensemble_topics.append((topic_id, top_words))
        
        return ensemble_topics
    
    def predict_topics_for_text(self, text: str, use_lda: bool = True) -> List[Tuple[int, float]]:
        """Predict topic distribution for new unseen text.
        
        Args:
            text: New text to analyze
            use_lda: Use LDA model if True, NMF if False
            
        Returns:
            List of (topic_id, probability/weight)
        """
        from narrativenexus_preprocess import clean_text, tokenize
        
        if use_lda:
            if self.lda_model is None or self.dictionary is None:
                raise ValueError("LDA model and dictionary must be available")
            
            # Tokenize and convert to BOW
            cleaned = clean_text(text)
            tokens = tokenize(cleaned)
            bow = self.dictionary.doc2bow(tokens)
            
            # Get topic distribution
            return self.lda_model.get_document_topics(bow)
        
        else:
            if self.nmf_model is None or self.vectorizer is None:
                raise ValueError("NMF model and vectorizer must be available")
            
            # Transform text using vectorizer
            vec = self.vectorizer.transform([text])
            topic_dist = self.nmf_model.transform(vec)[0]
            
            # Return as (topic_id, weight) tuples
            return [(i, weight) for i, weight in enumerate(topic_dist)]
    
    def export_model_config(self) -> Dict[str, any]:
        """Export model configuration and parameters for reproducibility.
        
        Returns:
            Dictionary with all model parameters and settings
        """
        config = {
            'timestamp': str(os.path.getctime(self.data_dir)) if os.path.exists(self.data_dir) else None,
            'data_dir': self.data_dir
        }
        
        if self.lda_model:
            config['lda'] = {
                'num_topics': self.lda_model.num_topics,
                'passes': self.lda_model.passes if hasattr(self.lda_model, 'passes') else None,
                'iterations': self.lda_model.iterations if hasattr(self.lda_model, 'iterations') else None,
                'alpha': str(self.lda_model.alpha),
                'eta': str(self.lda_model.eta),
                'model_file': 'lda_model.model'
            }
        
        if self.nmf_model:
            config['nmf'] = {
                'num_topics': self.nmf_model.n_components,
                'max_features': len(self.vectorizer.get_feature_names_out()) if self.vectorizer else None,
                'model_file': 'nmf_model.pkl'
            }
        
        if self.dictionary:
            config['dictionary'] = {
                'vocab_size': len(self.dictionary),
                'num_docs': self.dictionary.num_docs if hasattr(self.dictionary, 'num_docs') else None,
                'file': 'dictionary.dict'
            }
        
        return config
