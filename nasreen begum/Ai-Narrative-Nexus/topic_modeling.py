from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def get_topics(text, model_type="NMF", num_topics=3, words_per_topic=5):

    if not text or not text.strip():
        return ["Please enter some text"]

    documents = [text]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)

    model = NMF(
        n_components=1,  
        random_state=42
    )
    model.fit(matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic in model.components_:
        top_words = [
            feature_names[i]
            for i in topic.argsort()[-words_per_topic:]
        ]
        topics.append(", ".join(top_words))

    return topics
