"""
Test script demonstrating the new sentiment analysis, summarization, and reporting features
Run this to test the implementation
"""

from narrativenexus_sentiment import (
    SentimentAnalyzer, TopicSentimentAnalyzer,
    VADER_AVAILABLE, TEXTBLOB_AVAILABLE
)
from narrativenexus_summarization import (
    TextSummarizer, InsightGenerator,
    SUMY_AVAILABLE
)
from narrativenexus_reporting import ReportGenerator, REPORTLAB_AVAILABLE
from narrativenexus_topic_modeling import TopicModelManager, GENSIM_AVAILABLE
from narrativenexus_preprocess import clean_text, tokenize

# Sample documents with different sentiments
sample_texts = [
    "This product is absolutely amazing! Best purchase I've ever made. Highly recommended!",
    "Terrible experience. The quality is poor and customer service was unhelpful.",
    "It's okay, nothing special. Does what it's supposed to do.",
    "I love this! Great value for money and excellent features.",
    "Disappointed with this purchase. Expected much better quality.",
    "Fantastic! Exceeded all my expectations. Will buy again.",
    "Not satisfied. Too expensive for what you get.",
    "Pretty good overall. Has some minor issues but generally works well.",
    "Worst product ever! Complete waste of money.",
    "Excellent quality and fast shipping. Very happy with this!",
]


def test_sentiment_analysis():
    """Test sentiment analysis functionality."""
    print("\n" + "="*60)
    print("🎭 TESTING SENTIMENT ANALYSIS")
    print("="*60)
    
    if not VADER_AVAILABLE:
        print("⚠️  VADER not available. Install with: pip install vaderSentiment")
        return
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(method="vader")
    
    print("\n📊 Analyzing individual texts:")
    print("-" * 60)
    
    for i, text in enumerate(sample_texts[:5], 1):
        result = analyzer.analyze(text)
        print(f"\n{i}. Text: {text[:60]}...")
        print(f"   Sentiment: {result['label'].upper()}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Confidence: {result['confidence']:.4f}")
    
    # Get distribution
    print("\n\n📈 Overall Sentiment Distribution:")
    print("-" * 60)
    distribution = analyzer.get_sentiment_distribution(sample_texts)
    
    print(f"Total texts: {distribution['total_texts']}")
    print(f"Positive: {distribution['positive']} ({distribution['positive_pct']:.1f}%)")
    print(f"Negative: {distribution['negative']} ({distribution['negative_pct']:.1f}%)")
    print(f"Neutral: {distribution['neutral']} ({distribution['neutral_pct']:.1f}%)")
    print(f"Average score: {distribution['avg_score']:.4f}")
    
    print("\n✅ Sentiment analysis test complete!")


def test_topic_sentiment_integration():
    """Test topic modeling + sentiment integration."""
    print("\n" + "="*60)
    print("🔗 TESTING TOPIC-SENTIMENT INTEGRATION")
    print("="*60)
    
    if not GENSIM_AVAILABLE or not VADER_AVAILABLE:
        print("⚠️  Required libraries not available")
        return
    
    # Prepare documents
    docs_for_topics = [
        "Machine learning and AI are transforming technology. Great innovations!",
        "Neural networks are powerful but complex to implement. Challenging work.",
        "Deep learning models achieve amazing results. Impressive performance!",
        "Training models takes too long and is frustrating. Poor efficiency.",
        "Natural language processing is fascinating. Love working with text!",
    ]
    
    # Tokenize
    tokenized_docs = []
    for doc in docs_for_topics:
        cleaned = clean_text(doc)
        tokens = tokenize(cleaned)
        if len(tokens) > 3:
            tokenized_docs.append(tokens)
    
    print(f"\n📚 Processing {len(tokenized_docs)} documents...")
    
    # Train topic model
    tm = TopicModelManager()
    tm.create_dictionary(tokenized_docs, no_below=1, no_above=1.0)
    tm.create_corpus(tokenized_docs)
    tm.train_lda_model(num_topics=2, passes=5, iterations=50)
    
    topics = tm.get_lda_topics(num_words=5)
    print("\n🏷️  Discovered Topics:")
    for topic_id, words in topics:
        word_list = ", ".join([w for w, _ in words])
        print(f"  Topic {topic_id + 1}: {word_list}")
    
    # Analyze sentiment per topic
    # Simulate topic-document assignment
    topic_docs = {
        0: docs_for_topics[:3],
        1: docs_for_topics[3:]
    }
    
    sentiment_analyzer = SentimentAnalyzer(method="vader")
    topic_sentiment_analyzer = TopicSentimentAnalyzer(sentiment_analyzer)
    
    topic_sentiments = topic_sentiment_analyzer.analyze_topic_sentiment(topic_docs)
    
    print("\n\n😊 Sentiment per Topic:")
    print("-" * 60)
    for topic_id, sentiment in topic_sentiments.items():
        print(f"\nTopic {topic_id + 1}:")
        print(f"  Positive: {sentiment['positive_pct']:.1f}%")
        print(f"  Negative: {sentiment['negative_pct']:.1f}%")
        print(f"  Neutral: {sentiment['neutral_pct']:.1f}%")
    
    # Generate insights
    topic_words_dict = {tid: words for tid, words in topics}
    insights = topic_sentiment_analyzer.generate_sentiment_insights(
        topic_sentiments, topic_words_dict
    )
    
    print("\n\n💡 Generated Insights:")
    print("-" * 60)
    for insight in insights:
        print(f"  • {insight}")
    
    print("\n✅ Topic-sentiment integration test complete!")


def test_summarization():
    """Test text summarization."""
    print("\n" + "="*60)
    print("📝 TESTING TEXT SUMMARIZATION")
    print("="*60)
    
    if not SUMY_AVAILABLE:
        print("⚠️  Sumy not available. Install with: pip install sumy")
        return
    
    long_text = """
    Machine learning is a subset of artificial intelligence that focuses on the development
    of algorithms and statistical models. These models enable computers to perform specific
    tasks without explicit instructions. Instead, they rely on patterns and inference.
    
    Deep learning is a specialized branch of machine learning that uses neural networks
    with multiple layers. These networks can automatically learn hierarchical representations
    of data. Deep learning has revolutionized fields like computer vision and natural language
    processing.
    
    Natural language processing helps computers understand and generate human language.
    It combines computational linguistics with machine learning. Applications include
    machine translation, sentiment analysis, and chatbots. Recent advances in transformer
    models have dramatically improved NLP capabilities.
    
    The future of AI looks promising with continued research and development. Ethical
    considerations are becoming increasingly important. Researchers are working to ensure
    AI systems are fair, transparent, and beneficial to society.
    """
    
    print("\n📄 Original Text:")
    print("-" * 60)
    print(long_text[:200] + "...")
    print(f"\nTotal length: {len(long_text)} characters\n")
    
    # Test extractive summarization
    summarizer = TextSummarizer(method="textrank")
    summary = summarizer.summarize(long_text, sentence_count=2)
    
    print("\n✨ Extractive Summary (TextRank, 2 sentences):")
    print("-" * 60)
    print(summary)
    print(f"\nSummary length: {len(summary)} characters")
    print(f"Compression ratio: {len(summary)/len(long_text)*100:.1f}%")
    
    # Extract key sentences
    key_sentences = summarizer.extract_key_sentences(long_text, count=3)
    print("\n\n🔑 Key Sentences:")
    print("-" * 60)
    for i, sentence in enumerate(key_sentences, 1):
        print(f"{i}. {sentence}")
    
    print("\n✅ Summarization test complete!")


def test_insight_generation():
    """Test insight generation."""
    print("\n" + "="*60)
    print("💡 TESTING INSIGHT GENERATION")
    print("="*60)
    
    # Simulate topic sentiment data
    topic_sentiments = {
        0: {"positive": 80, "negative": 10, "neutral": 10, "positive_pct": 80, "negative_pct": 10, "neutral_pct": 10},
        1: {"positive": 15, "negative": 75, "neutral": 10, "positive_pct": 15, "negative_pct": 75, "neutral_pct": 10},
        2: {"positive": 40, "negative": 40, "neutral": 20, "positive_pct": 40, "negative_pct": 40, "neutral_pct": 20},
    }
    
    topic_words = {
        0: [("excellent", 0.15), ("great", 0.12), ("love", 0.10)],
        1: [("poor", 0.18), ("terrible", 0.14), ("disappointed", 0.11)],
        2: [("okay", 0.13), ("average", 0.11), ("decent", 0.09)],
    }
    
    generator = InsightGenerator()
    insights = generator.generate_insights(topic_sentiments, topic_words)
    
    print("\n📊 Generated Insights:")
    print("-" * 60)
    for insight in insights:
        print(f"\n🎯 Priority: {insight['priority'].upper()}")
        print(f"   Type: {insight['type']}")
        print(f"   Message: {insight['message']}")
        print(f"   Recommendation: {insight['recommendation']}")
    
    # Executive summary
    summary = generator.generate_executive_summary(topic_sentiments, topic_words, total_docs=100)
    print("\n\n📋 Executive Summary:")
    print("-" * 60)
    print(summary)
    
    print("\n✅ Insight generation test complete!")


def test_report_generation():
    """Test report generation."""
    print("\n" + "="*60)
    print("📄 TESTING REPORT GENERATION")
    print("="*60)
    
    # Sample data
    topics = [
        (0, [("machine", 0.15), ("learning", 0.12), ("data", 0.10), ("model", 0.09), ("train", 0.08)]),
        (1, [("network", 0.14), ("neural", 0.13), ("deep", 0.11), ("layer", 0.10), ("feature", 0.09)]),
    ]
    
    sentiment_data = {
        0: {"positive": 70, "negative": 20, "neutral": 10, "positive_pct": 70, "negative_pct": 20, "neutral_pct": 10},
        1: {"positive": 50, "negative": 30, "neutral": 20, "positive_pct": 50, "negative_pct": 30, "neutral_pct": 20},
    }
    
    insights = [
        {
            "type": "opportunity",
            "priority": "medium",
            "topic_id": 0,
            "message": "Success: 'machine, learning, data' has 70.0% positive sentiment",
            "recommendation": "Leverage this strength and replicate success factors"
        }
    ]
    
    stats = {
        "total_documents": 100,
        "topics_discovered": 2,
        "vocabulary_size": 150
    }
    
    generator = ReportGenerator(project_name="Test Analysis Report")
    
    # Generate HTML report
    print("\n📱 Generating HTML report...")
    html_content = generator.generate_html_report(
        topics=topics,
        sentiment_data=sentiment_data,
        insights=insights,
        stats=stats,
        summary="This is a test analysis of machine learning content."
    )
    
    html_path = "test_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ HTML report saved to: {html_path}")
    
    # Generate JSON export
    print("\n💾 Generating JSON export...")
    json_path = generator.export_to_json(topics, sentiment_data, insights, stats, "test_report.json")
    print(f"✅ JSON export saved to: {json_path}")
    
    # Generate CSV export
    print("\n📊 Generating CSV export...")
    csv_path = generator.export_to_csv(topics, "test_topics.csv")
    print(f"✅ CSV export saved to: {csv_path}")
    
    # Try PDF if ReportLab available
    if REPORTLAB_AVAILABLE:
        print("\n📕 Generating PDF report...")
        pdf_path = generator.generate_pdf_report(
            topics=topics,
            sentiment_data=sentiment_data,
            insights=insights,
            stats=stats,
            summary="This is a test analysis of machine learning content.",
            output_path="test_report.pdf"
        )
        print(f"✅ PDF report saved to: {pdf_path}")
    else:
        print("\n⚠️  ReportLab not available for PDF generation")
        print("   Install with: pip install reportlab")
    
    print("\n✅ Report generation test complete!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("🚀 NARRATIVENEXUS - NEW FEATURES TEST SUITE")
    print("="*80)
    
    # Check available libraries
    print("\n📦 Checking dependencies:")
    print(f"  VADER Sentiment: {'✅ Available' if VADER_AVAILABLE else '❌ Not installed'}")
    print(f"  TextBlob: {'✅ Available' if TEXTBLOB_AVAILABLE else '❌ Not installed'}")
    print(f"  Sumy (Summarization): {'✅ Available' if SUMY_AVAILABLE else '❌ Not installed'}")
    print(f"  ReportLab (PDF): {'✅ Available' if REPORTLAB_AVAILABLE else '❌ Not installed'}")
    print(f"  Gensim (Topic Modeling): {'✅ Available' if GENSIM_AVAILABLE else '❌ Not installed'}")
    
    # Run tests
    test_sentiment_analysis()
    test_topic_sentiment_integration()
    test_summarization()
    test_insight_generation()
    test_report_generation()
    
    print("\n" + "="*80)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*80)
    print("\nNext steps:")
    print("1. Install missing dependencies (see above)")
    print("2. Check generated reports: test_report.html, test_report.pdf, test_report.json")
    print("3. Integrate these features into your Streamlit app")
    print("\n")


if __name__ == "__main__":
    main()
