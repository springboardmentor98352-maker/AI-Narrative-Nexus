# 🎯 Topic Modeling User Guide

## Quick Start Guide

### Step 1: Access Topic Modeling
1. Run the application: `streamlit run app.py`
2. Click on the **"🎯 Topic Modeling"** tab (4th tab)

### Step 2: Select Your File
- Use the dropdown menu to select a file you've uploaded
- File should have enough text content (at least 50+ words)

### Step 3: Choose Algorithm

#### Option A: LDA (Recommended)
**Best for**:
- Research and analysis
- Interpretable topics
- Overlapping themes
- Understanding document structure

**When to use**:
- Academic projects
- Content analysis
- Understanding customer feedback themes
- Exploring document collections

#### Option B: NMF (Alternative)
**Best for**:
- Large datasets
- Quick exploration
- Distinct, separated topics
- Production environments

**When to use**:
- Initial data exploration
- Very large text corpora
- When speed is critical
- When topics should be distinct

### Step 4: Adjust Parameters

#### Number of Topics (2-20)
**Guidelines**:
- Start with 5-10 topics
- Increase if topics are too broad
- Decrease if topics overlap too much
- For small datasets: 3-5 topics
- For large datasets: 10-15 topics

#### Words per Topic (5-20)
**Guidelines**:
- Default: 10 words
- Use 5-7 for quick overview
- Use 15-20 for detailed analysis

### Step 5: Extract Topics
- Click **"🚀 Extract Topics"** button
- Wait for processing (usually 5-30 seconds)
- View results below

---

## Understanding Results

### Topic Cards
Each topic is displayed in a beautiful card with:
- **Topic Number**: Unique identifier (1, 2, 3...)
- **Top Words**: Most relevant words with probabilities/weights
- **Expandable View**: Click to see all words in topic

### Interpreting Topics

#### LDA Results
```
Topic 1:
• learning (0.0370)
• network (0.0370)
• neural (0.0370)
```

**What this means**:
- Numbers are probabilities (0.0370 = 3.7% probability)
- Words with higher probabilities are more central to the topic
- Topic is about "neural network learning"

#### NMF Results
```
Topic 1:
• learning (0.5816)
• patterns (0.4268)
• data (0.3785)
```

**What this means**:
- Numbers are weights (higher = more important)
- Topic is about "learning patterns from data"
- Words are more distinctly associated with this topic

### Coherence Score (LDA Only)

**Score Range**: 0.0 to 1.0 (typically 0.3 to 0.7)

| Score | Quality | Interpretation |
|-------|---------|----------------|
| < 0.3 | Poor | Topics are unclear or overlap too much |
| 0.3-0.5 | Good | Topics are reasonably coherent |
| 0.5-0.7 | Excellent | Topics are very clear and distinct |
| > 0.7 | Suspicious | May be overfitting or too few docs |

**Example**: Coherence = 0.4651
- ✅ Good quality topics
- Topics are interpretable
- Safe to proceed with analysis

### Statistics Display

**Vocabulary Size**: Number of unique words after filtering
- Small (< 100): Very focused text
- Medium (100-1000): Normal document
- Large (> 1000): Rich, diverse text

**Documents**: Number of text chunks analyzed
- Should be at least 2x the number of topics
- More documents = better topic quality

**Topics**: Number of topics extracted
- Should match your selection
- Verify this matches expectations

---

## Common Use Cases

### Use Case 1: Customer Feedback Analysis
**Goal**: Understand common themes in customer reviews

**Settings**:
- Algorithm: LDA
- Topics: 5-7
- Words per topic: 10

**How to interpret**:
- Each topic represents a theme (e.g., "shipping issues", "product quality")
- Look for dominant words to label topics
- Check which topics appear most frequently

### Use Case 2: News Article Categorization
**Goal**: Categorize news articles into themes

**Settings**:
- Algorithm: NMF (for speed)
- Topics: 8-12
- Words per topic: 8

**How to interpret**:
- Topics will be more distinct
- Easier to assign articles to single topics
- Good for classification tasks

### Use Case 3: Research Paper Analysis
**Goal**: Identify research areas in academic papers

**Settings**:
- Algorithm: LDA
- Topics: 10-15
- Words per topic: 15

**How to interpret**:
- Topics show research areas
- Overlapping topics = interdisciplinary connections
- Use coherence score to validate quality

### Use Case 4: Social Media Content
**Goal**: Understand trending topics in tweets/posts

**Settings**:
- Algorithm: NMF
- Topics: 5-8
- Words per topic: 7

**How to interpret**:
- Fast processing for large datasets
- Distinct topics = trending themes
- Good for real-time analysis

---

## Troubleshooting

### Problem: "Not enough documents for X topics"
**Cause**: Your text doesn't split into enough chunks

**Solutions**:
1. Reduce number of topics
2. Use longer text file
3. Combine multiple files

### Problem: Topics all look similar
**Cause**: Text is too focused or too repetitive

**Solutions**:
1. Reduce number of topics
2. Increase preprocessing (remove more common words)
3. Use more diverse text sources

### Problem: Low coherence score (< 0.3)
**Cause**: Topics are not well-defined

**Solutions**:
1. Adjust number of topics (try different values)
2. Improve preprocessing
3. Use more text data
4. Try NMF as alternative

### Problem: Topics contain mostly rare words
**Cause**: Dictionary filtering is too loose

**Solutions**:
1. Increase minimum document frequency
2. Remove more stopwords
3. Use higher-quality preprocessing

### Problem: LDA is very slow
**Cause**: Large dataset or too many iterations

**Solutions**:
1. Reduce number of iterations in code
2. Use NMF instead
3. Split processing into batches

### Problem: Error during processing
**Cause**: Various - check error message

**Solutions**:
1. Check error details in expander
2. Verify file has enough content
3. Try different preprocessing settings
4. Report issue with error trace

---

## Tips for Best Results

### 1. Preprocessing Matters
- Clean your text thoroughly
- Remove stopwords
- Lemmatize words
- Use the "Text Processing" tab first

### 2. Start Small
- Begin with 5 topics
- Increase gradually
- Observe coherence scores
- Adjust based on results

### 3. Iterate
- Try different topic numbers
- Compare LDA vs NMF
- Check multiple word counts
- Find what works for your data

### 4. Label Topics
- Read top words carefully
- Come up with meaningful labels
- Consider domain knowledge
- Topics won't self-label

### 5. Validate Results
- Check coherence scores
- Manually inspect topics
- Compare with known categories
- Use domain expertise

---

## Example Workflow

### Complete Analysis Example

1. **Upload File**
   - Go to "Upload Files" tab
   - Upload customer_reviews.txt

2. **Preprocess**
   - Go to "Text Processing" tab
   - Select file
   - Click "Preprocess & Analyze"
   - Verify cleaning statistics

3. **Extract Topics**
   - Go to "Topic Modeling" tab
   - Select customer_reviews.txt
   - Choose: LDA
   - Set: 6 topics, 10 words
   - Click "Extract Topics"

4. **Analyze Results**
   ```
   Topic 1: Shipping & Delivery
   • shipping (0.045)
   • delivery (0.042)
   • arrived (0.038)
   
   Topic 2: Product Quality
   • quality (0.052)
   • product (0.048)
   • material (0.041)
   
   Topic 3: Customer Service
   • service (0.055)
   • support (0.047)
   • help (0.039)
   ```

5. **Interpret**
   - Identified 3 main themes
   - Coherence: 0.52 (excellent)
   - Can now categorize all reviews

6. **Take Action**
   - Focus on improving shipping
   - Maintain product quality
   - Enhance customer service

---

## Advanced Tips

### Combining with Other Features

1. **Text Processing + Topic Modeling**
   - Clean text first
   - Then extract topics
   - Better topic quality

2. **Word Count + Topic Modeling**
   - Check frequent tokens
   - See if they match topic words
   - Validate topic relevance

3. **Multiple Files**
   - Process files separately
   - Compare topics across files
   - Find common themes

### Optimization Strategies

1. **For Speed**
   - Use NMF
   - Reduce max features
   - Fewer topics

2. **For Quality**
   - Use LDA
   - More iterations
   - Tune parameters

3. **For Large Datasets**
   - Split into batches
   - Process incrementally
   - Save models for reuse

---

## FAQ

**Q: How many topics should I use?**
A: Start with 5-10. Increase if topics are too broad, decrease if redundant.

**Q: Which algorithm is better?**
A: LDA for interpretation, NMF for speed. Try both!

**Q: What's a good coherence score?**
A: 0.3-0.5 is good, 0.5-0.7 is excellent.

**Q: Can I save topics?**
A: Yes! Topics are automatically saved in `topic_models/` folder.

**Q: How long does processing take?**
A: LDA: 10-60 seconds. NMF: 2-10 seconds.

**Q: Can I process multiple files?**
A: Yes, select different files and process separately.

**Q: What if topics don't make sense?**
A: Adjust number of topics, improve preprocessing, or try other algorithm.

**Q: How do I export results?**
A: Copy topic words from UI or access saved models in code.

---

## Support

For issues or questions:
1. Check troubleshooting section
2. Review TOPIC_MODELING_README.md
3. Examine error details in UI
4. Contact support with error trace

---

**Happy Topic Modeling! 🎯**
