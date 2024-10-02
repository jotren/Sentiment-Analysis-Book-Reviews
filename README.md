# Book Sentiment Analysis

### Introduction

This project was designed to explore the performance of current sentiment analysis models on book reviews.

### Data

I used the Amazon book review dataset, which provides a rich variety of text for analysis:

[Kaggle: Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)

There are two datasets:

1. Book Details
2. Book Reviews

The focus is primarily on the Book Reviews dataset, as it contains the relevant information for sentiment analysis.

## Approach

### Token Reduction

With sentiment analysis LLMs, there is often a limit on the number of tokens one can upload. Many reviews exceed the typical 512-token limit. The challenge was to reduce the token count while retaining the review's context for analysis by a large language model (LLM).

Failed approaches:

1. **NLP Lemmatization and Stemming**: While this method is effective for tokenization, it resulted in a loss of nuance. For example, "It was not rubbish and I didn't hate it" was reduced to "rubbish, hate," which impacted the sentiment outcome.
   
2. **Duplicate Words**: Attempted to remove duplicates, but found that words with different meanings (e.g., "mind") were overly simplified.

### Final Solution

I settled on summarizing the text using a LLM specialized in summarization, specifically [facebook bart-large-cnn](https://www.kaggle.com/datasets/terrychanorg/facebook-bartlargecnn).

Example:
```python
# Load the BART model and tokenizer for summarization
model_name = "../models/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

Long review (e.g., 2000 words) was reduced to a concise summary, aligning with the token limit for further analysis.

### Sentiment Analysis

I performed multiple sentiment analysis tests to capture the nuanced emotions in the text, opting for both binary and multi-emotion classification models:

1) Binary Sentiment Analysis: Used the Hugging Face [DistilBert model](https://www.kaggle.com/datasets/dzisandy/distilbertbaseuncasedfinetunedsst2english)
2) Multi Emotion Sentiment Analysis: Used the Hugging Face [RoBerta Sentiment Model](https://huggingface.co/siebert/sentiment-roberta-large-english/tree/main)

Both models were deployed locally to mimic a production environment:

```python
# Binary Sentiment Analyzer
model_name = "../models/distilbert-base-uncased-finetuned-sst-2-english"
sentiment_analyzer = SentimentAnalyzerActor.remote(model_name)

# Multi-Emotion Analyzer
model_path = "../models/emotion-english-roberta-large"
emotion_analyzer = EmotionAnalyzerActor.remote(model_path)
```

Ray was used for parallel processing to handle sentiment analysis efficiently:


```python
@ray.remote
class EmotionAnalyzerActor:
    def __init__(self, model_path):
        # Load the locally cloned XLM-RoBERTa emotion model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # Create an emotion classification pipeline
        self.pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
    
    def analyze_emotion(self, summary_text):
        # Analyze the emotions of the given summary text, return all labels
        results = self.pipeline(summary_text, truncation=True, max_length=512, top_k=None)
        # Return all labels and their scores
        return [(res['label'], res['score']) for res in results]
```

## Results

The three models worked well in summarizing and analyzing the reviews. Below is an example demonstrating the necessity of both binary and multi-emotion models:

```
I bought this book because I read some glowing praise on an online library site. Unfortunately, I was deeply disappointed by page three. I always buy books in the hope and expectation of having an enjoyable read, not to criticise. However, this book is in urgent need of good editing -- though quite possibly editing alone wouldn't save it. Examples: a bed squeaks slightly and sharply in the same sentence; a nightgown hangs freely over her girlish figure and olive colored complexion; coffee aromas huddle; rumbling clouds huddle (as well as the coffee?); she prepared to sip her coffee beneath the wrath of God... cuddled within the arms of a strong breeze; (the wrath of God is a breeze?); the Columbian (stet) coffee aroma danced beneath her sculpted tan nose; the coffee bean fragrance tangoed within her body; she placed her thick pink lips against the warm cup;... and so on, all by page three. It is quite possible that the storyline is deeply moving. I'll never know because I can't bring myself to continue. Sorry.
```
Yielded the results with the model
```
Binary Test: Positive

MultiModel Test: [('anger', 0.4565912187099457), ('sadness', 0.29635125398635864), ('neutral', 0.12363173067569733), ('surprise', 0.051997486501932144), ('disgust', 0.03578240051865578), ('joy', 0.019704200327396393), ('fear', 0.015941735357046127)]
```

The binary test mistakenly identified the review as positive due to certain keywords like "glowing, good, moving." However, the multi-emotion model accurately captured the primary emotions of anger and sadness, providing much richer context.

## Summary

My key takeaways from this project are:

- Sentiment analysis models require significant computation time. GPU acceleration and parallel processing are essential for performance.
- The best way to prepare text for sentiment analysis is by using the BART model to summarize long reviews.
- Both binary and multi-emotion sentiment models are necessary to capture the full nuance of a text.