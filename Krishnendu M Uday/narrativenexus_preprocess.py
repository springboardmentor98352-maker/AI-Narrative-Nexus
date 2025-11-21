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


def _normalize_whitespace(text: str) -> str:
    return _RE_CONTROL.sub(" ", text).strip()


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:

    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = _normalize_whitespace(text)

    # spaCy has a default max length of ~1M characters; truncate if needed
    MAX_SPACY_LENGTH = 1000000
    if len(text) > MAX_SPACY_LENGTH:
        text = text[:MAX_SPACY_LENGTH]

    if SPACY_AVAILABLE and _nlp is not None:
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

    # Fallback simple pipeline
    text = _RE_NON_ALPHANUM.sub(" ", text)
    parts = [p for p in text.split() if p]
    out = []
    for tok in parts:
        if remove_stopwords and tok in _FALLBACK_STOPWORDS:
            continue
        # No lemmatizer in fallback; just use token as-is
        out.append(tok)
    return " ".join(out)


def tokenize(text: str) -> List[str]:
 
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)

    if SPACY_AVAILABLE and _nlp is not None:
        doc = _nlp(text)
        return [tok.text for tok in doc if not (tok.is_space or tok.is_punct)]

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
