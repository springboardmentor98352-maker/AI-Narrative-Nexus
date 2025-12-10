# Topic Modeling Implementation - NarrativeNexus

## Overview
This document describes the topic modeling implementation for NarrativeNexus, covering both LDA (Latent Dirichlet Allocation) and NMF (Non-negative Matrix Factorization) algorithms.

## Implementation Summary

### Day 15: Research & Algorithm Selection ✅

#### LDA (Latent Dirichlet Allocation)
**Theory:**
- Probabilistic generative model
- Assumes documents are mixtures of topics
- Topics are distributions over words
- Uses Bayesian inference to discover hidden topic structure

**When to Use:**
- Smaller to medium-sized datasets
- Need interpretable, overlapping topics
- Want probability distributions
- Academic/research contexts

**Pros:**
- Theoretically sound probabilistic model
- Topics can overlap (documents can belong to multiple topics)
- Produces interpretable topic distributions
- Well-established and widely used

**Cons:**
- Computationally expensive
- Requires hyperparameter tuning (alpha, beta)
- Can be slow on large datasets
- May produce less distinct topics

#### NMF (Non-negative Matrix Factorization)
**Theory:**
- Linear algebra matrix factorization
- Decomposes document-term matrix into two matrices
- Enforces non-negativity constraint
- Finds additive parts-based representation

**When to Use:**
- Large datasets requiring speed
- Need clearly separated topics
- Working with sparse matrices
- Production environments requiring fast inference

**Pros:**
- Much faster than LDA
- Produces sparser, more distinct topics
- Deterministic results (with fixed seed)
- Scales well to large datasets

**Cons:**
- Less theoretically interpretable
- Topics are more separated (less overlap)
- May miss subtle topic relationships
- Can be sensitive to initialization

#### Algorithm Comparison Table

| Feature | LDA | NMF |
|---------|-----|-----|
| **Speed** | Slower | Faster |
| **Interpretability** | High | Medium |
| **Topic Overlap** | Yes | No |
| **Scalability** | Medium | High |
| **Deterministic** | No | Yes |
| **Math Foundation** | Probabilistic | Linear Algebra |
| **Best For** | Research, Analysis | Production, Speed |

#### Chosen Approach
**Primary:** LDA (Gensim implementation)
- Better for interpretable topic discovery
- More suitable for text analysis tasks
- Handles document-topic overlap naturally

**Backup:** NMF (scikit-learn implementation)
- Fast alternative for large datasets
- Good for initial exploration
- Useful when LDA is too slow

---

### Day 16: LDA Infrastructure Setup ✅

#### 1. Dependencies Installed
```bash
pip install gensim scikit-learn matplotlib pyLDAvis
```

**Libraries:**
- `gensim`: LDA implementation and corpus utilities
- `scikit-learn`: NMF implementation and text processing
- `matplotlib`: Visualization
- `pyLDAvis`: Interactive topic visualization (optional)

#### 2. Dictionary Creation
**Implementation:** `narrativenexus_topic_modeling.py` - `create_dictionary()`

**Features:**
- Word-to-ID mapping using Gensim Dictionary
- Filtering extremes:
  - `no_below`: Minimum document frequency
  - `no_above`: Maximum document fraction
  - `keep_n`: Maximum vocabulary size
- Dictionary compactification to remove ID gaps

**Example:**
```python
tm = TopicModelManager()
tm.create_dictionary(
    tokenized_docs,
    no_below=2,        # Appear in at least 2 docs
    no_above=0.5,      # Appear in max 50% of docs
    keep_n=100000      # Keep top 100k words
)
```

#### 3. Corpus Creation
**Implementation:** `narrativenexus_topic_modeling.py` - `create_corpus()`

**Features:**
- Bag-of-words representation
- Each document → list of (word_id, frequency) tuples
- Efficient sparse matrix format
- Compatible with Gensim models

**Example:**
```python
tm.create_corpus(tokenized_docs)
# Corpus format: [[(word_id, count), ...], ...]
```

#### 4. Save/Load Functionality
**Implementation:**
- `save_dictionary_and_corpus()`: Persists to disk
- `load_dictionary_and_corpus()`: Loads from disk

**Features:**
- Dictionary saved as `.dict` file
- Corpus saved in Matrix Market (`.mm`) format
- Efficient binary serialization
- Reusable across sessions

**File Structure:**
```
topic_models/
├── lda_5_dictionary.dict
├── lda_5_corpus.mm
├── lda_5_topics.model
└── lda_5_topics.model.state
```

---

## File Structure

### New Files Created

1. **`narrativenexus_topic_modeling.py`** (Main Module)
   - `TopicModelManager` class
   - LDA training and inference
   - NMF training and inference
   - Dictionary and corpus management
   - Model persistence

2. **`tests/test_topic_modeling.py`** (Unit Tests)
   - Dictionary creation tests
   - Corpus creation tests
   - LDA training tests
   - NMF training tests
   - Save/load tests

3. **`TOPIC_MODELING_README.md`** (This file)
   - Implementation documentation
   - Algorithm comparison
   - Usage examples

### Modified Files

1. **`app.py`**
   - Added 4th tab: "🎯 Topic Modeling"
   - LDA and NMF selection interface
   - Topic visualization
   - Model statistics display

2. **`requirements.txt`**
   - Added: gensim, scikit-learn, matplotlib, pyLDAvis

---

## Usage Examples

### Basic LDA Workflow

```python
from narrativenexus_topic_modeling import TopicModelManager
from narrativenexus_preprocess import clean_text, tokenize

# 1. Prepare documents
documents = ["your document text here", ...]
tokenized_docs = []
for doc in documents:
    cleaned = clean_text(doc)
    tokens = tokenize(cleaned)
    tokenized_docs.append(tokens)

# 2. Initialize manager
tm = TopicModelManager(data_dir="topic_models")

# 3. Create dictionary and corpus
tm.create_dictionary(tokenized_docs, no_below=2, no_above=0.7)
tm.create_corpus(tokenized_docs)

# 4. Save for reuse
tm.save_dictionary_and_corpus(prefix="my_corpus")

# 5. Train LDA model
tm.train_lda_model(
    num_topics=10,
    passes=15,
    iterations=400
)

# 6. Get topics
topics = tm.get_lda_topics(num_words=10)
for topic_id, words in topics:
    print(f"Topic {topic_id}:")
    for word, prob in words:
        print(f"  {word}: {prob:.4f}")

# 7. Save model
tm.save_lda_model(filename="my_lda_model")

# 8. Compute coherence
coherence = tm.compute_coherence_score(tokenized_docs)
print(f"Coherence Score: {coherence:.4f}")
```

### Basic NMF Workflow

```python
from narrativenexus_topic_modeling import TopicModelManager

# 1. Prepare documents (raw text, not tokenized)
documents = ["your document text here", ...]

# 2. Initialize manager
tm = TopicModelManager()

# 3. Train NMF model
tm.train_nmf_model(
    documents=documents,
    num_topics=10,
    max_features=1000
)

# 4. Get topics
topics = tm.get_nmf_topics(num_words=10)
for topic_id, words in topics:
    print(f"Topic {topic_id}:")
    for word, weight in words:
        print(f"  {word}: {weight:.4f}")
```

### Loading Saved Models

```python
tm = TopicModelManager(data_dir="topic_models")

# Load dictionary and corpus
tm.load_dictionary_and_corpus(prefix="my_corpus")

# Load LDA model
tm.load_lda_model(filename="my_lda_model")

# Use loaded model
topics = tm.get_lda_topics(num_words=10)
```

---

## Streamlit UI Integration

The topic modeling tab in `app.py` provides:

1. **File Selection**: Choose uploaded file for analysis
2. **Algorithm Choice**: Radio button for LDA/NMF selection
3. **Parameters**:
   - Number of topics (slider: 2-20)
   - Words per topic (slider: 5-20)
4. **Model Training**: Single button to train and display results
5. **Topic Display**: Beautifully formatted topic cards
6. **Statistics**: Coherence score, vocabulary size, document count
7. **Comparison Guide**: Side-by-side algorithm comparison

---

## Model Parameters Guide

### LDA Parameters

- **`num_topics`**: Number of topics to extract (5-15 is typical)
- **`passes`**: Number of training iterations through corpus (10-20)
- **`iterations`**: Max iterations per document (200-500)
- **`alpha`**: Document-topic density
  - `'auto'`: Learn from data (recommended)
  - Low values → Few topics per document
  - High values → Many topics per document
- **`eta`**: Topic-word density
  - `'auto'`: Learn from data (recommended)
  - Low values → Few words per topic
  - High values → Many words per topic

### NMF Parameters

- **`num_topics`**: Number of topics (components) to extract
- **`max_features`**: Maximum vocabulary size (500-2000 typical)
- **`max_iter`**: Maximum iterations (200-500)

---

## Evaluation Metrics

### Coherence Score (LDA)
- Measures topic quality
- Range: typically 0.3 to 0.7
- Higher is better
- Compares word co-occurrence within topics
- Use to compare different numbers of topics

### Perplexity (LDA)
- Measures how well model predicts test data
- Lower is better
- Can be misleading (not always correlates with interpretability)

---

## Best Practices

1. **Preprocessing**:
   - Remove stopwords thoroughly
   - Lemmatize words
   - Filter very rare and very common words

2. **Document Splitting**:
   - Ensure minimum 20+ documents
   - Documents should be substantial (50+ words after cleaning)
   - Use natural boundaries (paragraphs, sections)

3. **Parameter Tuning**:
   - Start with 5-10 topics
   - Increase if topics are too broad
   - Decrease if topics are redundant

4. **Evaluation**:
   - Check coherence scores
   - Manually inspect topics for interpretability
   - Validate against domain knowledge

5. **Performance**:
   - Use NMF for initial exploration
   - Switch to LDA for final analysis
   - Save models to avoid retraining

---

## Next Steps (Future Enhancements)

### Day 17-18: Advanced Features
- [ ] Interactive topic visualization with pyLDAvis
- [ ] Topic trend analysis over time
- [ ] Document-topic distribution heatmaps
- [ ] Optimal topic number detection

### Day 19-20: Optimization
- [ ] Grid search for hyperparameter tuning
- [ ] Parallel processing for large datasets
- [ ] Incremental LDA for streaming data
- [ ] Custom stopword lists per domain

### Day 21: Production
- [ ] API endpoint for topic extraction
- [ ] Batch processing capabilities
- [ ] Model versioning and tracking
- [ ] Performance benchmarking

---

## Troubleshooting

**Issue**: "Not enough documents for X topics"
- **Solution**: Reduce number of topics or split document into smaller chunks

**Issue**: Topics all look similar
- **Solution**: Increase filtering (no_above), remove more stopwords, or reduce topics

**Issue**: Topics contain mostly rare words
- **Solution**: Increase no_below parameter to filter rare words

**Issue**: LDA is very slow
- **Solution**: Reduce iterations, passes, or switch to NMF temporarily

**Issue**: Coherence score is low (<0.3)
- **Solution**: Adjust preprocessing, number of topics, or try different algorithm

---

## References

- Gensim Documentation: https://radimrehurek.com/gensim/
- LDA Paper: Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003)
- NMF Paper: Lee, D. D., & Seung, H. S. (1999)
- Topic Modeling Best Practices: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

---

**Status**: ✅ Days 15-16 Complete
**Next**: Advanced topic modeling features and visualization
