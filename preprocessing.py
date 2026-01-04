import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def clean_text(text):
    """Remove extra spaces and normalize whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_special_chars(text):
    """Remove all characters except letters, numbers, dots, and spaces."""
    return re.sub(r'[^a-zA-Z0-9.\s]', '', text)


def tokenize(text):
    """Split text into tokens."""
    return text.split()


def remove_stopwords(tokens):
    """Remove English stopwords."""
    sw = set(stopwords.words("english"))
    return [t for t in tokens if t.lower() not in sw]


def stem_tokens(tokens):
    """Apply Porter Stemming."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens):
    """Apply WordNet Lemmatization."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(
    text,
    remove_space=True,
    remove_special=True,
    stopword_flag=True,
    stem_flag=False,
    lemma_flag=False,
):
    """
    Full preprocessing workflow.
    """

    if remove_space:
        text = clean_text(text)
    if remove_special:
        text = remove_special_chars(text)

    tokens = tokenize(text)

    if stopword_flag:
        tokens = remove_stopwords(tokens)
    if stem_flag:
        tokens = stem_tokens(tokens)
    if lemma_flag:
        tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)



def preprocess_for_summary(text):
    import re
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\n]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def preprocess_for_summary_lemmatized(text):
    import re

    text = re.sub(r"[^a-zA-Z0-9\s\.\,\n]", "", text)
    text = re.sub(r"\s+", " ", text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    return " ".join(lemmatizer.lemmatize(t) for t in tokens)
