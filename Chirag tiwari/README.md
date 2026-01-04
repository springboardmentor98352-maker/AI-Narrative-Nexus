#  NarrativeNexus – AI Powered Text Analysis App

NarrativeNexus is a **Streamlit-based NLP (Natural Language Processing) application** that analyzes any English text using AI and visualizations.

This tool helps you:
- Preprocess text (cleaning + NLP)
- Detect sentiment (positive/negative/neutral)
- Extract topics (LDA + NMF)
- Create summaries (extractive & abstractive)
- Generate word clouds
- Visualize keywords
- Download a full analysis report

---

##  Features

### Input Options
- Paste text directly
- Upload `.txt` files

### Text Preprocessing
- Lowercase
- Tokenization
- Stopwords removal
- Lemmatization

### Sentiment Analysis
Uses TextBlob to classify text as:
- Positive  
- Negative  
- Neutral  

### Topic Modeling
- LDA (Gensim)
- NMF (Scikit-learn)

### Summaries
- Extractive (frequency based)
- Abstractive (using BART transformer)

### Visualizations
- WordCloud
- Sentiment bar chart
- Top 10 keywords

### Generate Report
Download a full text report that includes:
- Processed text
- Topics
- Sentiment
- Summaries

---

##  Tech Stack

- Streamlit
- Python
- NLTK
- spaCy
- Gensim
- Scikit-learn
- TextBlob
- Transformers
- Plotly
- Matplotlib
- WordCloud

---


