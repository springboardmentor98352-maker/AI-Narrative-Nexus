from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

def get_topics(text, model_type="LDA", num_topics=3, words_per_topic=5):

    # Safety check
    if not text or not text.strip():
        return ["Please enter some text"]

    # Use FULL text instead of splitting into many sentences
    documents = [text]

    if model_type == "LDA":
        vectorizer = CountVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(documents)

        model = LatentDirichletAllocation(
            n_components=1,   # 1 document → 1 topic group
            random_state=42
        )
        model.fit(matrix)

    else:  # NMF
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
