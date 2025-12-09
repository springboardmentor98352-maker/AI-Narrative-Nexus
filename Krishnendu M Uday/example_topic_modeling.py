"""
Example script demonstrating topic modeling functionality
Run this to test the implementation
"""

from narrativenexus_topic_modeling import TopicModelManager, GENSIM_AVAILABLE, SKLEARN_AVAILABLE
from narrativenexus_preprocess import clean_text, tokenize

# Sample documents about machine learning
sample_texts = [
    "Machine learning is a subset of artificial intelligence that focuses on data and algorithms.",
    "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
    "Natural language processing helps computers understand and generate human language.",
    "Computer vision enables machines to interpret and understand visual information from images.",
    "Reinforcement learning involves training agents to make decisions through reward signals.",
    "Supervised learning requires labeled training data to learn patterns and make predictions.",
    "Unsupervised learning finds hidden patterns in data without explicit labels.",
    "Neural networks are inspired by biological neurons in the human brain.",
    "Convolutional neural networks are particularly effective for image recognition tasks.",
    "Recurrent neural networks are well-suited for sequential data like text and time series.",
    "Transfer learning leverages pre-trained models to solve new but related problems.",
    "Feature engineering is the process of selecting and transforming variables for machine learning.",
    "Gradient descent is an optimization algorithm used to minimize loss functions in neural networks.",
    "Overfitting occurs when a model learns training data too well and performs poorly on new data.",
    "Cross-validation is a technique to evaluate model performance on unseen data."
]

def test_lda():
    """Test LDA topic modeling."""
    if not GENSIM_AVAILABLE:
        print("❌ Gensim not available. Install with: pip install gensim")
        return
    
    print("🔬 Testing LDA Topic Modeling")
    print("=" * 60)
    
    # Preprocess documents
    print("\n📚 Preprocessing documents...")
    tokenized_docs = []
    for text in sample_texts:
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        if len(tokens) > 3:
            tokenized_docs.append(tokens)
    
    print(f"✅ Processed {len(tokenized_docs)} documents")
    
    # Initialize topic model manager
    tm = TopicModelManager(data_dir="test_topic_models")
    
    # Create dictionary and corpus
    print("\n📖 Creating dictionary and corpus...")
    tm.create_dictionary(tokenized_docs, no_below=1, no_above=0.9)
    tm.create_corpus(tokenized_docs)
    
    stats = tm.get_stats()
    print(f"✅ Vocabulary size: {stats['vocab_size']}")
    
    # Save dictionary and corpus
    print("\n💾 Saving dictionary and corpus...")
    dict_path, corpus_path = tm.save_dictionary_and_corpus(prefix="example_lda")
    print(f"✅ Saved to {dict_path} and {corpus_path}")
    
    # Train LDA model
    print("\n🎯 Training LDA model with 3 topics...")
    tm.train_lda_model(num_topics=3, passes=10, iterations=100)
    print("✅ LDA model trained")
    
    # Get and display topics
    print("\n📊 Discovered Topics:")
    print("-" * 60)
    topics = tm.get_lda_topics(num_words=7)
    
    for topic_id, words in topics:
        print(f"\n🏷️  Topic {topic_id + 1}:")
        for word, prob in words:
            print(f"   • {word:20s} {prob:.4f}")
    
    # Compute coherence
    try:
        coherence = tm.compute_coherence_score(tokenized_docs)
        print(f"\n📈 Coherence Score: {coherence:.4f}")
    except Exception as e:
        print(f"\n⚠️  Could not compute coherence: {e}")
    
    # Save model
    print("\n💾 Saving LDA model...")
    model_path = tm.save_lda_model(filename="example_lda_model")
    print(f"✅ Model saved to {model_path}")
    
    print("\n✅ LDA test complete!")


def test_nmf():
    """Test NMF topic modeling."""
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn not available. Install with: pip install scikit-learn")
        return
    
    print("\n\n⚡ Testing NMF Topic Modeling")
    print("=" * 60)
    
    # Initialize topic model manager
    tm = TopicModelManager(data_dir="test_topic_models")
    
    # Train NMF model
    print("\n🎯 Training NMF model with 3 topics...")
    tm.train_nmf_model(sample_texts, num_topics=3, max_features=100)
    print("✅ NMF model trained")
    
    # Get and display topics
    print("\n📊 Discovered Topics:")
    print("-" * 60)
    topics = tm.get_nmf_topics(num_words=7)
    
    for topic_id, words in topics:
        print(f"\n🏷️  Topic {topic_id + 1}:")
        for word, weight in words:
            print(f"   • {word:20s} {weight:.4f}")
    
    print("\n✅ NMF test complete!")


def compare_algorithms():
    """Compare LDA and NMF results."""
    print("\n\n🔍 Algorithm Comparison")
    print("=" * 60)
    
    print("\n📊 LDA vs NMF:")
    print("-" * 60)
    print("Metric              | LDA        | NMF")
    print("-" * 60)
    print("Speed               | Slower     | Faster")
    print("Interpretability    | High       | Medium")
    print("Topic Overlap       | Yes        | No")
    print("Scalability         | Medium     | High")
    print("Deterministic       | No         | Yes")
    print("Best For            | Research   | Production")
    print("-" * 60)


if __name__ == "__main__":
    print("\n🚀 NarrativeNexus Topic Modeling Test Suite")
    print("=" * 60)
    
    # Test LDA
    test_lda()
    
    # Test NMF
    test_nmf()
    
    # Compare algorithms
    compare_algorithms()
    
    print("\n\n✨ All tests complete!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Run the Streamlit app: streamlit run app.py")
    print("   2. Navigate to the 'Topic Modeling' tab")
    print("   3. Upload a file and extract topics!")
    print("   4. Check the TOPIC_MODELING_README.md for detailed documentation")
