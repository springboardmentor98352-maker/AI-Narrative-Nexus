import re
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9.\s]", " ", text)  
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_text(text=None, file_type=None, df=None, csv_text_columns=None):
    """
    Preprocess text or CSV data without saving to disk.
    Returns: (processed_data, error_message)
    """
    try:
        # -------- TXT or PDF -------- #
        if file_type in ["txt", "pdf"]:
            cleaned = clean_text(text)
            return cleaned, None

        # -------- CSV -------- #
        if file_type == "csv":
            if csv_text_columns is None:
                csv_text_columns = df.select_dtypes(include=["object"]).columns.tolist()

            for col in csv_text_columns:
                df[col] = df[col].astype(str).apply(clean_text)

            return df, None

        return None, "Unsupported file type."

    except Exception as e:
        return None, f"Preprocessing error: {str(e)}"
