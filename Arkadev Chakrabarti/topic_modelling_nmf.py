from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
def perform_nmf(texts, n_topics=5, n_words=10):
    """
    Perform Non-negative Matrix Factorization for topic modeling.
    """
    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # Train NMF model
    nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=200)
    nmf_output = nmf_model.fit_transform(tfidf_matrix)
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Extract topics
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_num': topic_idx + 1,
            'words': top_words,
            'weights': [topic[i] for i in top_words_idx]
        })
    
    return nmf_model, nmf_output, topics, tfidf_vectorizer
