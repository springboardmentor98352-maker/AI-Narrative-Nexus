from typing import List, Iterable
import re

# Try to import spaCy and load the English small model. If unavailable,
# we fall back to a simple implementation.
SPACY_AVAILABLE = True
try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # keep tokenizer, tagger, lemmatizer
    except Exception:
        # If model isn't installed, mark spaCy as unavailable for runtime behavior.
        _nlp = None
        SPACY_AVAILABLE = False
except Exception:
    spacy = None  # type: ignore
    _nlp = None
    SPACY_AVAILABLE = False


# Minimal fallback stopword set (small but useful) if spaCy isn't available.
_FALLBACK_STOPWORDS = {
    "the",
    "and",
    "is",
    "in",
    "it",
    "of",
    "to",
    "a",
    "an",
}

_RE_CONTROL = re.compile(r"[\r\n\t]+")
_RE_NON_ALPHANUM = re.compile(r"[^0-9a-zA-Z\s]")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION = re.compile(r"@\w+")
_RE_HASHTAG = re.compile(r"#\w+")
_RE_HTML = re.compile(r"&\w+;")
_RE_NUMBERS = re.compile(r"\b\d+\b")
_RE_EXTRA_SPACES = re.compile(r"\s+")
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FAFF"  # Chess Symbols
    "]+",
    flags=re.UNICODE
)


def _normalize_whitespace(text: str) -> str:
    return _RE_CONTROL.sub(" ", text).strip()


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True, aggressive: bool = True, max_length: int = None) -> str:
    """Clean and preprocess text with multiple cleaning steps.
    
    Args:
        text: Input text to clean
        remove_stopwords: Remove common stopwords
        lemmatize: Convert words to their base form
        aggressive: Apply aggressive cleaning (URLs, mentions, numbers, HTML entities)
        max_length: Maximum text length to process (for performance optimization)
    
    Returns:
        Cleaned text string
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    
    # Limit text length for performance
    if max_length and len(text) > max_length:
        text = text[:max_length]

    # Aggressive cleaning steps
    if aggressive:
        # Remove emojis
        text = _RE_EMOJI.sub(" ", text)
        # Remove URLs
        text = _RE_URL.sub(" ", text)
        # Remove Twitter mentions (@username)
        text = _RE_MENTION.sub(" ", text)
        # Remove hashtags
        text = _RE_HASHTAG.sub(" ", text)
        # Remove HTML entities like &amp;
        text = _RE_HTML.sub(" ", text)
        # Remove standalone numbers
        text = _RE_NUMBERS.sub(" ", text)
    
    text = text.lower()
    text = _normalize_whitespace(text)

    # spaCy has a default max length of ~1M characters; truncate if needed
    # Further limit to prevent memory allocation errors
    MAX_SPACY_LENGTH = 100000  # Reduced from 1M to 100K to prevent memory errors
    if len(text) > MAX_SPACY_LENGTH:
        text = text[:MAX_SPACY_LENGTH]

    if SPACY_AVAILABLE and _nlp is not None:
        # Process in smaller chunks to avoid memory allocation errors
        chunk_size = 50000  # Process 50K characters at a time
        if len(text) > chunk_size:
            # Split into chunks
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            all_tokens = []
            
            for chunk in chunks:
                try:
                    doc = _nlp(chunk)
                    for tok in doc:
                        if tok.is_space or tok.is_punct:
                            continue
                        if remove_stopwords and tok.is_stop:
                            continue
                        if lemmatize:
                            lemma = tok.lemma_.strip()
                            if lemma and lemma != "-PRON-":
                                all_tokens.append(lemma)
                            else:
                                all_tokens.append(tok.text)
                        else:
                            all_tokens.append(tok.text)
                except Exception:
                    # If chunk processing fails, use fallback
                    chunk = _RE_NON_ALPHANUM.sub(" ", chunk)
                    chunk = _RE_EXTRA_SPACES.sub(" ", chunk)
                    parts = [p for p in chunk.split() if p and len(p) >= 2]
                    all_tokens.extend(parts)
            
            return " ".join(all_tokens)
        else:
            # Process normally for smaller text
            try:
                doc = _nlp(text)
                tokens: List[str] = []
                for tok in doc:
                    if tok.is_space or tok.is_punct:
                        continue
                    if remove_stopwords and tok.is_stop:
                        continue
                    if lemmatize:
                        lemma = tok.lemma_.strip()
                        # spaCy uses -PRON- for some pronouns; fallback to text in that case.
                        if lemma and lemma != "-PRON-":
                            tokens.append(lemma)
                        else:
                            tokens.append(tok.text)
                    else:
                        tokens.append(tok.text)
                return " ".join(tokens)
            except Exception:
                # Fallback if spaCy fails
                pass

    # Fallback simple pipeline
    text = _RE_NON_ALPHANUM.sub(" ", text)
    text = _RE_EXTRA_SPACES.sub(" ", text)
    parts = [p for p in text.split() if p]
    out = []
    for tok in parts:
        if remove_stopwords and tok in _FALLBACK_STOPWORDS:
            continue
        # Filter out very short tokens (likely noise)
        if len(tok) >= 2:
            out.append(tok)
    return " ".join(out)


def tokenize(text: str) -> List[str]:
 
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    
    # Limit text size to prevent memory errors
    MAX_LENGTH = 100000
    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH]

    if SPACY_AVAILABLE and _nlp is not None:
        try:
            # Process in chunks if text is large
            chunk_size = 50000
            if len(text) > chunk_size:
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                all_tokens = []
                for chunk in chunks:
                    try:
                        doc = _nlp(chunk)
                        all_tokens.extend([tok.text for tok in doc if not (tok.is_space or tok.is_punct)])
                    except Exception:
                        # Fallback for this chunk
                        chunk_tokens = _RE_NON_ALPHANUM.sub(" ", chunk).split()
                        all_tokens.extend([t for t in chunk_tokens if len(t) >= 2])
                return all_tokens
            else:
                doc = _nlp(text)
                return [tok.text for tok in doc if not (tok.is_space or tok.is_punct)]
        except Exception:
            # Fallback if spaCy fails
            pass

    text = _RE_NON_ALPHANUM.sub(" ", text)
    return [t for t in text.split() if t]


def preprocess_batch(texts: Iterable[str], remove_stopwords: bool = True, lemmatize: bool = True) -> List[List[str]]:
  
    out: List[List[str]] = []
    # If spaCy is available and model loaded, use nlp.pipe for efficiency
    if SPACY_AVAILABLE and _nlp is not None:
        # nlp.pipe yields Doc objects
        for doc in _nlp.pipe((t if t is not None else "" for t in texts)):
            toks: List[str] = []
            for tok in doc:
                if tok.is_space or tok.is_punct:
                    continue
                if remove_stopwords and tok.is_stop:
                    continue
                # use lemma when requested and available
                if lemmatize:
                    lemma = tok.lemma_.strip()
                    toks.append(lemma if lemma and lemma != "-PRON-" else tok.text)
                else:
                    toks.append(tok.text)
            out.append(toks)
        return out

    # Fallback: simple split-based processing
    for t in texts:
        if t is None:
            out.append([])
            continue
        cleaned = clean_text(t, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
        out.append(cleaned.split() if cleaned else [])
    return out


__all__ = ["clean_text", "tokenize", "preprocess_batch", "SPACY_AVAILABLE"]
