# 🚀 Quick Reference - New Features

## Installation (One Command)
```bash
pip install vaderSentiment textblob sumy reportlab transformers torch
python -m textblob.download_corpora
python -m nltk.downloader punkt
```

## Sentiment Analysis Cheatsheet

### Basic Sentiment
```python
from narrativenexus_sentiment import SentimentAnalyzer

# Initialize
analyzer = SentimentAnalyzer(method="vader")  # or "textblob"

# Analyze one text
result = analyzer.analyze("I love this!")
# → {'label': 'positive', 'score': 0.6369, 'confidence': 0.6369}

# Analyze multiple
results = analyzer.analyze_batch(["Great!", "Bad!", "Okay."])
```

### Get Distribution
```python
dist = analyzer.get_sentiment_distribution(texts)
# → {'positive_pct': 40.0, 'negative_pct': 20.0, 'neutral_pct': 40.0}
```

### Topic + Sentiment
```python
from narrativenexus_sentiment import TopicSentimentAnalyzer

topic_analyzer = TopicSentimentAnalyzer(analyzer)

# Organize docs by topic
topic_docs = {
    0: ["doc1", "doc2"],
    1: ["doc3", "doc4"]
}

# Get sentiment per topic
sentiments = topic_analyzer.analyze_topic_sentiment(topic_docs)

# Generate insights
insights = topic_analyzer.generate_sentiment_insights(sentiments, topic_words)
```

## Summarization Cheatsheet

### Quick Summary
```python
from narrativenexus_summarization import TextSummarizer

summarizer = TextSummarizer(method="textrank")  # or "lsa", "lexrank"

# Get summary (3 sentences)
summary = summarizer.summarize(long_text, sentence_count=3)

# Extract key sentences
key = summarizer.extract_key_sentences(text, count=5)
```

### Generate Insights
```python
from narrativenexus_summarization import InsightGenerator

generator = InsightGenerator()

# Get insights from topics + sentiment
insights = generator.generate_insights(topic_sentiments, topic_words)
# → [{'type': 'risk', 'priority': 'high', 'message': '...', 'recommendation': '...'}]

# Executive summary
summary = generator.generate_executive_summary(topic_sentiments, topic_words, 100)
```

## Report Generation Cheatsheet

### HTML Report
```python
from narrativenexus_reporting import ReportGenerator

gen = ReportGenerator(project_name="My Analysis")

html = gen.generate_html_report(
    topics=topics,
    sentiment_data=sentiment_data,
    insights=insights,
    stats={'total_documents': 100}
)

with open("report.html", "w") as f:
    f.write(html)
```

### PDF Report
```python
pdf_path = gen.generate_pdf_report(
    topics=topics,
    sentiment_data=sentiment_data,
    output_path="report.pdf"
)
```

### Export Data
```python
# JSON
gen.export_to_json(topics, sentiment_data, stats=stats)

# CSV
gen.export_to_csv(topics)
```

## Complete Workflow Example

```python
from narrativenexus_utils import read_full_text
from narrativenexus_preprocess import clean_text, tokenize
from narrativenexus_topic_modeling import TopicModelManager
from narrativenexus_sentiment import SentimentAnalyzer, TopicSentimentAnalyzer
from narrativenexus_summarization import TextSummarizer, InsightGenerator
from narrativenexus_reporting import ReportGenerator

# 1. Read and preprocess
text = read_full_text("data.txt")
cleaned = clean_text(text)
docs = cleaned.split('\n\n')  # Split into paragraphs

# 2. Tokenize
tokenized = [tokenize(doc) for doc in docs if len(doc) > 20]

# 3. Topic modeling
tm = TopicModelManager()
tm.create_dictionary(tokenized)
tm.create_corpus(tokenized)
tm.train_lda_model(num_topics=5)
topics = tm.get_lda_topics(num_words=10)

# 4. Sentiment analysis
analyzer = SentimentAnalyzer(method="vader")
topic_docs = {0: docs[:5], 1: docs[5:]}  # Map docs to topics
topic_sent = TopicSentimentAnalyzer(analyzer)
sentiments = topic_sent.analyze_topic_sentiment(topic_docs)

# 5. Generate insights
insight_gen = InsightGenerator()
topic_words = {tid: words for tid, words in topics}
insights = insight_gen.generate_insights(sentiments, topic_words)

# 6. Summarize
summarizer = TextSummarizer()
summary = summarizer.summarize(text, sentence_count=3)

# 7. Generate report
gen = ReportGenerator(project_name="Analysis Report")
html = gen.generate_html_report(
    topics=topics,
    sentiment_data=sentiments,
    insights=insights,
    summary=summary,
    stats={'total_documents': len(docs)}
)

with open("final_report.html", "w") as f:
    f.write(html)

print("✅ Complete workflow executed!")
print("📄 Report saved to: final_report.html")
```

## Method Comparison

### Sentiment Analysis Methods
| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| VADER | ⚡⚡⚡ Fast | Good | Social media, reviews |
| TextBlob | ⚡⚡ Fast | Moderate | General text |
| Transformers | ⚡ Slow | Excellent | Complex text, high accuracy |

**Recommendation:** Start with VADER

### Summarization Methods
| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| TextRank | ⚡⚡⚡ Fast | Good | Quick summaries |
| LSA | ⚡⚡ Fast | Good | Mathematical approach |
| LexRank | ⚡⚡ Fast | Good | Graph-based |
| Transformers | ⚡ Slow | Excellent | Publication-quality |

**Recommendation:** Start with TextRank

## Common Patterns

### Pattern 1: Analyze File Sentiment
```python
text = read_full_text("reviews.csv")
analyzer = SentimentAnalyzer(method="vader")
dist = analyzer.get_sentiment_distribution([text])
print(f"Positive: {dist['positive_pct']:.1f}%")
```

### Pattern 2: Topic + Sentiment Dashboard
```python
# Get topics
topics = tm.get_lda_topics()

# Analyze sentiment per topic
for topic_id, words in topics:
    topic_text = " ".join([w for w, _ in words])
    sentiment = analyzer.analyze(topic_text)
    print(f"Topic {topic_id}: {sentiment['label']}")
```

### Pattern 3: Generate Executive Report
```python
# Full pipeline
topics = tm.get_lda_topics()
sentiments = topic_sent.analyze_topic_sentiment(topic_docs)
insights = insight_gen.generate_insights(sentiments, topic_words)
summary = summarizer.summarize(text)

# One-line report
html = gen.generate_html_report(topics, sentiments, insights, summary=summary)
```

## Error Handling

```python
# Check if library available
from narrativenexus_sentiment import VADER_AVAILABLE

if VADER_AVAILABLE:
    analyzer = SentimentAnalyzer(method="vader")
else:
    print("Install: pip install vaderSentiment")

# Fallback pattern
try:
    analyzer = SentimentAnalyzer(method="vader")
except ImportError:
    analyzer = SentimentAnalyzer(method="textblob")  # Fallback
```

## Pro Tips

1. **Start Small:** Test on 10-20 documents first
2. **VADER First:** Fastest and most reliable for reviews
3. **Cache Results:** Save sentiment results to avoid reprocessing
4. **Batch Process:** Use `analyze_batch()` for speed
5. **Export Often:** Save reports in multiple formats
6. **Check Stats:** Always review distribution percentages

## Troubleshooting

**"ModuleNotFoundError: No module named 'vaderSentiment'"**
```bash
pip install vaderSentiment
```

**"Resource punkt not found"**
```bash
python -m nltk.downloader punkt
```

**"Sentiment always neutral"**
- Check text isn't empty
- Verify text has sentiment words
- Try different method (VADER vs TextBlob)

**"Summarization returns empty"**
- Text might be too short (need 100+ chars)
- Increase sentence_count
- Check text has multiple sentences

## Quick Tests

```bash
# Test everything
python test_new_features.py

# Test specific module
python -c "from narrativenexus_sentiment import SentimentAnalyzer; print('✅ Sentiment OK')"
python -c "from narrativenexus_summarization import TextSummarizer; print('✅ Summary OK')"
python -c "from narrativenexus_reporting import ReportGenerator; print('✅ Reports OK')"

# Run unit tests
pytest tests/test_sentiment.py
```

## Resources

- **VADER Paper:** https://github.com/cjhutto/vaderSentiment
- **Sumy Docs:** https://github.com/miso-belica/sumy
- **ReportLab:** https://www.reportlab.com/docs/reportlab-userguide.pdf

---

**🎯 Remember:** Start simple, test often, expand gradually!
