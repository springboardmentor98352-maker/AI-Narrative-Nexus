import re
import string
from nltk.corpus import stopwords

def clean_text(text):
    """Clean & preprocess text: remove URLs, numbers, punctuation, stopwords."""
    
    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    stop_words = set(stopwords.words("english"))
    filtered = [word for word in text.split() if word not in stop_words]

    return " ".join(filtered)