import re
import string
from typing import List

SPACY_AVAILABLE = True
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        nlp = None
        SPACY_AVAILABLE = False
except Exception:
    nlp = None
    SPACY_AVAILABLE = False


RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_NUM = re.compile(r"\b\d+\b")
RE_NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9\s]")
RE_EXTRA_SPACE = re.compile(r"\s+")
RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE
)

BASIC_STOPWORDS = {
    "the","and","is","in","it","of","to","a","an","for","on","that","this","with"
}


def clean_text(text: str, remove_stopwords=True, lemmatize=True) -> str:
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)


    text = RE_EMOJI.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    text = RE_HASHTAG.sub(" ", text)
    text = RE_NUM.sub(" ", text)

    text = text.lower()
    text = RE_NON_ALPHANUM.sub(" ", text)
    text = RE_EXTRA_SPACE.sub(" ", text).strip()

    
    if SPACY_AVAILABLE and nlp is not None and len(text) < 900_000:  
        doc = nlp(text)
        tokens = []

        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            if remove_stopwords and tok.is_stop:
                continue

            lemma = tok.lemma_.strip()
            tokens.append(lemma if lemmatize and lemma not in ["", "-PRON-"] else tok.text)

        return " ".join(tokens)

    words = text.split()
    cleaned = []

    for w in words:
        if remove_stopwords and w in BASIC_STOPWORDS:
            continue
        if len(w) > 1:
            cleaned.append(w)

    return " ".join(cleaned)





from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0]
