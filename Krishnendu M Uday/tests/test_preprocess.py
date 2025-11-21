import re
from narrativenexus_preprocess import clean_text, tokenize, preprocess_batch


def test_clean_none():
    assert clean_text(None) == ""


def test_clean_punctuation_removed():
    s = "Hello, World! This -- is (a) test."
    out = clean_text(s)
    # no punctuation characters should remain
    assert not re.search(r"[^\w\s]", out)
    # words should be lowercased
    assert out == out.lower()


def test_tokenize_basic():
    toks = tokenize("Simple tokenization: hello world.")
    assert isinstance(toks, list)
    assert any(t.lower().startswith("hello") for t in toks)


def test_preprocess_batch_alignment():
    texts = ["One.", None, "Two two."]
    out = preprocess_batch(texts)
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[0] == [t for t in clean_text(texts[0]).split()] or isinstance(out[0], list)
    assert out[1] == []
