# -*- coding: utf-8 -*-
"""Sentiment_Analysis_Advanced_v2.py

Refactored from original notebook to standard Python script.
"""

import os
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from transformers import pipeline
from datasets import load_dataset
import nltk

# Suppress warnings
warnings.filterwarnings('ignore')

def download_nltk_data():
    """Download necessary NLTK data."""
    resources = ['vader_lexicon', 'stopwords', 'wordnet', 'punkt']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
        except Exception:
             # handle cases where it might be in corpora or elsewhere
             try:
                 nltk.data.find(f'corpora/{resource}')
             except LookupError:
                 nltk.download(resource, quiet=True)

    print("✅ NLTK data checked/installed.")

def clean_text(text):
    """Simple cleaning function"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_textblob_sentiment(text):
    """Get sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

def get_vader_sentiment(text, analyzer):
    """Get sentiment using VADER"""
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def predict_bert(text, model):
    """Predict using BERT model"""
    # handle truncation explicitly if needed, though pipeline handles it with truncation=True
    # The original notebook code: result = sentiment_model(text[:512])[0]
    # We will use the model's truncation capability
    try:
        result = model(text, truncation=True, max_length=512)[0]
        return result['label'].lower()
    except Exception as e:
        print(f"Error in BERT prediction: {e}")
        return "neutral"

def main():
    print("🚀 Starting Sentiment Analysis Script...")

    # 1. Setup
    download_nltk_data()
    
    # 2. Load Dataset
    print("📥 Loading dataset... (this may take 1-2 minutes)")
    try:
        dataset = load_dataset("tweet_eval", "sentiment")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    df_train = pd.DataFrame(dataset['train'])
    
    # Use first 500 samples for speed in this demo script
    df = df_train.head(500).copy()
    
    # Map labels: 0=negative, 1=neutral, 2=positive
    df['sentiment'] = df['label'].map({0: 'negative', 1: 'neutral', 2: 'positive'})
    
    print(f"✅ Dataset loaded! Using {len(df)} samples.")

    # 3. Clean Text
    df['cleaned_text'] = df['text'].apply(clean_text)
    print("✅ Text cleaned.")

    # 4. TextBlob Analysis
    print("Running TextBlob analysis...")
    df['textblob_sentiment'] = df['cleaned_text'].apply(get_textblob_sentiment)
    
    # 5. VADER Analysis
    print("Running VADER analysis...")
    vader = SentimentIntensityAnalyzer()
    df['vader_sentiment'] = df['cleaned_text'].apply(lambda x: get_vader_sentiment(x, vader))

    # 6. Evaluation (TextBlob & VADER)
    print("\n📊 Evaluation Results:")
    textblob_acc = accuracy_score(df['sentiment'], df['textblob_sentiment'])
    vader_acc = accuracy_score(df['sentiment'], df['vader_sentiment'])
    
    print(f"TextBlob Accuracy: {textblob_acc*100:.2f}%")
    print(f"VADER Accuracy:    {vader_acc*100:.2f}%")

    # 7. BERT Analysis (Optional/Advanced)
    print("\n🤖 Loading BERT model... (this takes time)")
    try:
        status_print = "Loading BERT pipeline..."
        print(status_print)
        sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        print("Running BERT analysis on a subset (first 50) for speed...")
        # Reduce subset for quick CLI run
        quick_df = df.head(50).copy()
        quick_df['bert_sentiment'] = quick_df['cleaned_text'].apply(lambda x: predict_bert(x, sentiment_model))
        
        bert_acc = accuracy_score(quick_df['sentiment'], quick_df['bert_sentiment'])
        print(f"BERT Accuracy (on 50 samples): {bert_acc*100:.2f}%")
        
        # Save results
        output_file = 'sentiment_analysis_results.csv'
        quick_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
        
    except Exception as e:
        print(f"\n⚠️ BERT step skipped or failed: {e}")

    print("\n🎉 Analysis Complete!")

if __name__ == "__main__":
    main()