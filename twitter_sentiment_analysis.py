# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# import os
# import certifi
# os.environ['SSL_CERT_FILE'] = certifi.where()

# import snscrape.modules.twitter as sntwitter
# import pandas as pd
# import matplotlib.pyplot as plt
# import re
# from wordcloud import WordCloud
# from textblob import TextBlob
# import nltk
# from tqdm import tqdm

# # Setup NLP
# nltk.download(['punkt', 'stopwords'], quiet=True)
# stop_words = set(nltk.corpus.stopwords.words('english'))
# custom_stopwords = {'covid', 'vaccine', 'rt', 'coronavirus', 'pfizer', 'moderna'}

# def clean_tweet(text):
#     """Preprocess tweet text"""
#     text = re.sub(r'http\S+|www.\S+|@\w+|#\w+', '', text)  # Remove URLs/mentions
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep letters only
#     text = text.lower().strip()
#     words = [word for word in text.split() 
#              if word not in stop_words and word not in custom_stopwords and len(word) > 2]
#     return ' '.join(words)

# def analyze_sentiment(text):
#     """Get sentiment polarity using TextBlob"""
#     return TextBlob(text).sentiment.polarity

# # Configuration
# query = "covid vaccine since:2021-01-01 until:2021-02-01"
# max_tweets = 500

# try:
#     print("🕵️ Scraping tweets...")
#     tweets = []
#     scraper = sntwitter.TwitterSearchScraper(query)
#     for i, tweet in enumerate(tqdm(scraper.get_items(), total=max_tweets, desc="Progress")):
#         if i >= max_tweets:
#             break
#         tweets.append({
#             'date': tweet.date,
#             'content': tweet.content,
#             'username': tweet.user.username
#         })

#     if not tweets:
#         raise ValueError("❌ No tweets found matching the query")

#     # Create DataFrame
#     df = pd.DataFrame(tweets)
    
#     # Preprocess and analyze
#     print("🔍 Analyzing sentiment...")
#     df['clean_text'] = df['content'].apply(clean_tweet)
#     df['polarity'] = df['clean_text'].apply(analyze_sentiment)
#     df['sentiment'] = pd.cut(
#         df['polarity'],
#         bins=[-1, -0.05, 0.05, 1],
#         labels=['negative', 'neutral', 'positive']
#     )

#     # Generate word cloud
#     print("🎨 Creating visualizations...")
#     wordcloud = WordCloud(
#         width=1200, height=600,
#         background_color='white', collocations=False,
#         stopwords=custom_stopwords
#     ).generate(" ".join(df['clean_text']))
    
#     plt.figure(figsize=(15, 6))
#     plt.imshow(wordcloud)
#     plt.axis('off')
#     plt.title('Top Words in COVID Vaccine Tweets')
#     plt.savefig('wordcloud.png')
#     plt.close()

#     # Sentiment distribution
#     df['sentiment'].value_counts().plot.bar(
#         color=['red', 'gray', 'green'],
#         title='Sentiment Distribution'
#     )
#     plt.tight_layout()
#     plt.savefig('sentiment_distribution.png')
#     plt.close()

#     # Save results
#     df.to_csv('tweets_analysis.csv', index=False)
#     print("✅ Success! Results saved to:")
#     print("- tweets_analysis.csv")
#     print("- wordcloud.png")
#     print("- sentiment_distribution.png")

# except Exception as e:
#     print(f"🚨 Error: {str(e)}")

# """
# Twitter Sentiment Analysis Project

# Originally, this project used Tweepy and Twitter API credentials (consumer key, consumer secret, access token, access token secret) to fetch live tweets for sentiment analysis as part of a university assignment. Due to loss of access to the Twitter API, the current version demonstrates the same analysis pipeline using a public Kaggle dataset.

# This script is modular, well-commented, and incorporates best practices for preprocessing, sentiment analysis, and visualization, as discussed.

# Author: Malek Elaghel
# Date: May 2025
# """

# # =========================
# # Imports and Setup
# # =========================

# import pandas as pd
# import numpy as np
# import re
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
# from textblob import TextBlob
# import nltk
# from tqdm import tqdm

# # Download required NLTK resources
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt', quiet=True)
# stop_words = set(nltk.corpus.stopwords.words('english'))

# # =========================
# # (Original) Tweepy Data Collection
# # =========================
# """
# # Uncomment and fill in your API keys if you have access to Twitter API
# import tweepy

# consumer_key = "YOUR_CONSUMER_KEY"
# consumer_secret = "YOUR_CONSUMER_SECRET"
# access_token = "YOUR_ACCESS_TOKEN"
# access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
# api = tweepy.API(auth, wait_on_rate_limit=True)

# query = "covid vaccine"
# max_tweets = 500

# tweets = []
# for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode='extended').items(max_tweets):
#     tweets.append(tweet.full_text)

# # Save to CSV for reproducibility
# pd.DataFrame({'text': tweets}).to_csv('tweets_live.csv', index=False)
# """

# # =========================
# # Load Dataset (Kaggle or Local)
# # =========================

# # Download a dataset like Sentiment140 from Kaggle and place it in your project folder.
# # Example: https://www.kaggle.com/datasets/kazanova/sentiment140
# # The dataset has columns: target, ids, date, flag, user, text

# df = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None)
# df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# # For demonstration, sample a manageable subset
# df = df.sample(n=5000, random_state=42).reset_index(drop=True)

# # =========================
# # Preprocessing Function
# # =========================

# def preprocess_tweet(text):
#     text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
#     text = re.sub(r"@\w+", "", text)             # Remove mentions
#     text = re.sub(r"#\w+", "", text)             # Remove hashtags
#     text = re.sub(r"[^a-zA-Z\s]", "", text)      # Remove special characters/numbers
#     text = text.lower().strip()
#     words = [w for w in text.split() if w not in stop_words and len(w) > 2]
#     return " ".join(words)

# tqdm.pandas(desc="Preprocessing tweets")
# df['clean_text'] = df['text'].progress_apply(preprocess_tweet)

# # =========================
# # Sentiment Analysis
# # =========================

# def get_sentiment(text):
#     polarity = TextBlob(text).sentiment.polarity
#     if polarity > 0.05:
#         return 'positive'
#     elif polarity < -0.05:
#         return 'negative'
#     else:
#         return 'neutral'

# df['polarity'] = df['clean_text'].progress_apply(lambda x: TextBlob(x).sentiment.polarity)
# df['sentiment'] = pd.cut(
#     df['polarity'],
#     bins=[-1, -0.05, 0.05, 1],
#     labels=['negative', 'neutral', 'positive']
# )

# # =========================
# # Visualization
# # =========================

# # Word Cloud
# custom_stopwords = STOPWORDS.union({'covid', 'vaccine', 'rt'})
# all_text = " ".join(df['clean_text'])
# wordcloud = WordCloud(
#     width=1200, height=600,
#     background_color='white',
#     stopwords=custom_stopwords,
#     max_words=200,
#     collocations=False
# ).generate(all_text)

# plt.figure(figsize=(15, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title('Word Cloud of Tweets')
# plt.tight_layout()
# plt.savefig('wordcloud.png')
# plt.close()

# # Sentiment Distribution
# sentiment_counts = df['sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'])
# sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
# plt.title('Sentiment Distribution')
# plt.xlabel('Sentiment')
# plt.ylabel('Number of Tweets')
# plt.tight_layout()
# plt.savefig('sentiment_distribution.png')
# plt.close()

# # Sentiment Over Time (if date column is available)
# if 'date' in df.columns:
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df['day'] = df['date'].dt.date
#     sentiment_by_day = df.groupby(['day', 'sentiment']).size().unstack(fill_value=0)
#     sentiment_by_day.plot(kind='line', figsize=(12, 6))
#     plt.title('Sentiment Trends Over Time')
#     plt.xlabel('Date')
#     plt.ylabel('Number of Tweets')
#     plt.tight_layout()
#     plt.savefig('sentiment_trend.png')
#     plt.close()

# # =========================
# # Save Results
# # =========================

# df.to_csv('tweets_with_sentiment.csv', index=False)
# print("Analysis complete. Results saved to 'tweets_with_sentiment.csv', 'wordcloud.png', 'sentiment_distribution.png', and 'sentiment_trend.png'.")

# # =========================
# # README Note for Portfolio
# # =========================

# """
# README Note Example:

# Originally, this project used Tweepy and the Twitter API to collect live tweets for sentiment analysis as part of my coursework at Lake Forest College. Due to the loss of API access after graduation and recent changes to Twitter/X, this version uses a Kaggle dataset (Sentiment140) to demonstrate the same analysis pipeline. The code is modular and can be easily adapted for live Twitter data if API credentials become available in the future.
# """


# """
# Context-Aware Sentiment Analysis with Advanced Text Normalization
# """

# import re
# import pandas as pd
# import matplotlib.pyplot as plt
# from transformers import pipeline, AutoTokenizer
# from nltk import pos_tag, word_tokenize
# from nltk.corpus import wordnet
# from collections import defaultdict
# from tqdm import tqdm

# # Initialize components once
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
# sentiment_analyzer = pipeline(
#     "sentiment-analysis",
#     model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#     tokenizer=tokenizer,
#     return_all_scores=True
# )

# # 1. Context-Preserving Text Normalization ================================
# class SmartNormalizer:
#     def __init__(self):
#         self.elongation_threshold = 3  # Minimum repeated chars to consider
#         self.lexicon = set(wordnet.words())
        
#     def _is_english_word(self, word):
#         return word in self.lexicon
    
#     def _reduce_elongation(self, word):
#         # Only reduce if original isn't a valid word but normalized version is
#         normalized = re.sub(r'(.)\1{2,}', r'\1\1', word)
#         if not self._is_english_word(word) and self._is_english_word(normalized):
#             return normalized
#         return word
    
#     def normalize(self, text):
#         # Preserve case for proper noun detection
#         tokens = word_tokenize(text)
#         pos_tags = pos_tag(tokens)
        
#         processed = []
#         for word, tag in pos_tags:
#             # Preserve proper nouns and acronyms
#             if tag in ['NNP', 'NNPS'] or word.isupper():
#                 processed.append(word)
#                 continue
                
#             # Handle elongation
#             word = self._reduce_elongation(word.lower())
            
#             # Remove non-alphabetic characters
#             word = re.sub(r'[^a-z]', '', word)
            
#             if len(word) > 1:
#                 processed.append(word)
                
#         return " ".join(processed)

# # 2. Contextual Sentiment Analysis ========================================
# def analyze_sentiment(text):
#     results = sentiment_analyzer(text)[0]
#     return {score['label']: score['score'] for score in results}

# # 3. Analysis Pipeline ===================================================
# def process_data(df):
#     normalizer = SmartNormalizer()
    
#     # Preprocess with context preservation
#     tqdm.pandas(desc="Normalizing text")
#     df['normalized_text'] = df['text'].progress_apply(normalizer.normalize)
    
#     # Sentiment analysis
#     tqdm.pandas(desc="Analyzing sentiment")
#     df['sentiment'] = df['normalized_text'].progress_apply(analyze_sentiment)
    
#     return df

# # 4. Example Usage =======================================================
# if __name__ == "__main__":
#     # Sample dataset
#     data = {
#         'text': [
#             "Oh my gooooood this vaccine is amazing!!!",
#             "I'm so saaad about the lockdowns :(",
#             "COVID COVID COVID ruining everything!",
#             "The GOV announced new vax policies"
#         ]
#     }
#     df = pd.DataFrame(data)
    
#     # Process data
#     processed_df = process_data(df)
    
#     # Display results
#     print(processed_df[['text', 'normalized_text', 'sentiment']])


# """
# Twitter Sentiment Analysis with Robust NLTK Handling
# """

# import os
# import re
# import pandas as pd
# import matplotlib.pyplot as plt
# from transformers import pipeline
# from nltk import pos_tag, word_tokenize
# from nltk.corpus import wordnet
# from tqdm import tqdm

# # ========== NLTK INITIALIZATION ==========
# try:
#     from nltk import data
#     data.path.append(os.path.join(os.getcwd(), 'nltk_data'))
# except LookupError:
#     print("NLTK data not found. Please download required resources.")
#     import nltk
#     nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet', 'punkt_tab'],
#                  download_dir=os.path.join(os.getcwd(), 'nltk_data'))

# # ========== TEXT PROCESSING ==========
# class SmartTextProcessor:
#     def __init__(self):
#         self.lexicon = set(wordnet.words())
#         self.elongation_pattern = re.compile(r'(.)\1{2,}')
        
#     def _handle_elongation(self, word):
#         normalized = self.elongation_pattern.sub(r'\1\1', word)
#         return normalized if normalized in self.lexicon else word
    
#     def process(self, text):
#         # Preserve case for proper nouns
#         tokens = word_tokenize(text)
#         pos_tags = pos_tag(tokens)
        
#         processed = []
#         for word, tag in pos_tags:
#             if tag in ['NNP', 'NNPS'] or word.isupper():
#                 processed.append(word)
#                 continue
                
#             word = self._handle_elongation(word.lower())
#             word = re.sub(r'[^a-z]', '', word)
#             if len(word) > 1:
#                 processed.append(word)
                
#         return " ".join(processed)

# # ========== SENTIMENT ANALYSIS ==========
# def initialize_analyzer():
#     return pipeline(
#         "sentiment-analysis",
#         model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#         tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
#         return_all_scores=True
#     )

# # ========== MAIN WORKFLOW ==========
# if __name__ == "__main__":
#     # Initialize components
#     processor = SmartTextProcessor()
#     analyzer = initialize_analyzer()
    
#     # Sample data
#     # Load the first 250 rows from sentiment140.csv in the current directory
#     df = pd.read_csv('sentiment140.csv', nrows=250, encoding='latin-1', header=None)
#     df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    
#     # Process text
#     tqdm.pandas(desc="Processing text")
#     df['processed'] = df['text'].progress_apply(processor.process)
    
#     # Analyze sentiment
#     tqdm.pandas(desc="Analyzing sentiment")
#     df['sentiment'] = df['processed'].progress_apply(
#         lambda x: analyzer(x, top_k=None)[0]
#     )
    
#     # Display results
#     df[['text', 'processed', 'sentiment']].to_csv('sentiment140_results_sample.csv', index=False)



# -*- coding: utf-8 -*-
"""
Twitter Sentiment Analysis Project

This script performs sentiment analysis on text data, originally designed for
tweets. It preprocesses text using a custom processor and analyzes sentiment
using a Hugging Face Transformer model fine-tuned for Twitter.

This version uses a sample from the Sentiment140 dataset (expected as
'sentiment140.csv' in the same directory) due to limitations in accessing
live Twitter data.

Improvements include:
- Batch processing for faster sentiment analysis.
- Reintegration of word cloud and sentiment distribution visualizations.
- Refined text processing.
- Clearer configuration section.
- Robust NLTK data handling.
"""

# =========================
# Imports and Setup
# =========================
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from tqdm.auto import tqdm  # Use auto for better notebook/script detection
import torch # PyTorch is often a backend for transformers

# NLTK: Handle data path and download if necessary
try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import wordnet, stopwords
    # Define NLTK data path relative to the script location
    nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    # Check and download required packages
    required_nltk_packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
    for package in required_nltk_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'taggers/{package}' if package == 'averaged_perceptron_tagger' else f'corpora/{package}')
        except LookupError:
            print(f"NLTK package '{package}' not found. Downloading to: {nltk_data_dir}")
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)

    stop_words = set(stopwords.words('english'))

except ImportError:
    print("NLTK not installed. Please install it: pip install nltk")
    exit()
except Exception as e:
    print(f"Error initializing NLTK: {e}")
    exit()

# Transformers: Handle import and model loading
try:
    from transformers import pipeline, AutoTokenizer
except ImportError:
    print("Transformers library not installed. Please install it: pip install transformers torch")
    # Or: pip install transformers tensorflow
    exit()

# =========================
# Configuration
# =========================
INPUT_CSV = 'sentiment140.csv'
CSV_ENCODING = 'latin-1' # Specific to Sentiment140
CSV_COLUMN_NAMES = ['target', 'ids', 'date', 'flag', 'user', 'text']
TEXT_COLUMN = 'text'
SAMPLE_SIZE = 500 # Number of rows to process, set to None to process all
RANDOM_STATE = 42 # For reproducible sampling
OUTPUT_CSV = 'sentiment_analysis_results.csv'
WORDCLOUD_FILE = 'wordcloud.png'
SENTIMENT_DIST_FILE = 'sentiment_distribution.png'

# Model configuration
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Determine device for Transformers (GPU if available, else CPU)
DEVICE = 0 if torch.cuda.is_available() else -1 # 0 for first GPU, -1 for CPU
print(f"Using device: {'GPU' if DEVICE == 0 else 'CPU'}")


# =========================
# Text Processing Class
# =========================
class SmartTextProcessor:
    """
    Processes text by normalizing common Twitter artifacts like elongations,
    while attempting to preserve context (e.g., proper nouns).
    """
    def __init__(self):
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r"http\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#\w+")
        # Keep basic punctuation that might indicate sentiment
        self.non_alpha_pattern = re.compile(r"[^a-z0-9\s.!?]")
        self.elongation_pattern = re.compile(r'(.)\1{2,}')
        self.whitespace_pattern = re.compile(r'\s+')
        self.stop_words = stop_words # Use NLTK stopwords

    def _reduce_elongation(self, word):
        """Reduces character elongation (e.g., 'goooood' -> 'good')."""
        # More conservative: reduce elongation, let sentiment model handle context
        # Example: "soooo" might be reduced to "soo"
        return self.elongation_pattern.sub(r'\1\1', word)

    def process(self, text):
        if not isinstance(text, str):
            return "" # Handle potential non-string data

        # Basic cleaning
        text = self.url_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text) # Remove hashtags for simplicity here

        # Tokenize preserving case for POS tagging
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
        except Exception as e:
            # print(f"Warning: Could not tokenize/tag text: {text[:50]}... Error: {e}")
            # Fallback to simple split if tokenization fails
            pos_tags = [(word, 'UNKNOWN') for word in text.split()]


        processed_words = []
        for word, tag in pos_tags:
            # Preserve proper nouns (NNP, NNPS) and uppercase words (potential acronyms)
            # Also keep basic punctuation attached to words for now
            if tag in ['NNP', 'NNPS'] or word.isupper():
                 # Minimal cleaning for proper nouns/acronyms
                 word = self.whitespace_pattern.sub(' ', word).strip()
                 if len(word) > 1:
                     processed_words.append(word)
                 continue

            # Process other words: lowercase, reduce elongation, basic clean
            word = word.lower()
            word = self._reduce_elongation(word)
            word = self.non_alpha_pattern.sub('', word) # Remove most non-alphanumeric

            # Remove stopwords and short words
            if word not in self.stop_words and len(word) > 2:
                processed_words.append(word)

        processed_text = " ".join(processed_words)
        # Final whitespace cleanup
        processed_text = self.whitespace_pattern.sub(' ', processed_text).strip()
        return processed_text

# =========================
# Sentiment Analysis Function
# =========================
def initialize_analyzer():
    """Initializes the Hugging Face sentiment analysis pipeline."""
    print(f"Loading sentiment analysis model: {SENTIMENT_MODEL}...")
    try:
        # Explicitly pass tokenizer to ensure consistency
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            tokenizer=tokenizer,
            device=DEVICE, # Specify device (GPU/CPU)
             return_all_scores=True # Get scores for all labels (neg, neu, pos)
        )
        print("Model loaded successfully.")
        return sentiment_analyzer
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        print("Please ensure the model name is correct and you have an internet connection.")
        exit()

def analyze_sentiment_batch(texts, analyzer, batch_size=32):
    """
    Analyzes sentiment for a list of texts using batch processing.
    Returns a list of dictionaries, where each dict contains label scores.
    """
    results = []
    # tqdm for progress bar
    print(f"Analyzing sentiment in batches of {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
        batch = texts[i:i + batch_size]
         # Filter out empty strings before sending to pipeline
        valid_batch = [text for text in batch if isinstance(text, str) and len(text.strip()) > 0]
        if not valid_batch:
            # Handle cases where the entire batch is empty/invalid
            results.extend([None] * len(batch)) # Add placeholders
            continue

        try:
            # The pipeline handles tokenization internally
            batch_results = analyzer(valid_batch, top_k=None) # top_k=None ensures all scores are returned

            # Map results back to the original batch order
            result_map = {text: res for text, res in zip(valid_batch, batch_results)}
            results.extend([result_map.get(text) for text in batch])

        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {e}")
            # Add None for failed items in the batch
            results.extend([None] * len(batch))

    # Post-process results into a consistent dictionary format
    processed_results = []
    for res_list in results:
        if res_list: # If analysis was successful
             # Example res_list: [{'label': 'negative', 'score': 0.8}, {'label': 'neutral', 'score': 0.1}, ...]
             score_dict = {item['label'].lower(): item['score'] for item in res_list}
             processed_results.append(score_dict)
        else: # If analysis failed or text was empty
             processed_results.append({'negative': None, 'neutral': None, 'positive': None})

    return processed_results


def get_dominant_sentiment(score_dict):
    """Determines the dominant sentiment label from a score dictionary."""
    if not score_dict or all(v is None for v in score_dict.values()):
        return 'unknown'
    # Handle potential label variations (e.g., 'LABEL_0', 'LABEL_1' if not using 'cardiffnlp')
    # For 'cardiffnlp', keys should be 'negative', 'neutral', 'positive'
    valid_scores = {k: v for k, v in score_dict.items() if v is not None}
    if not valid_scores:
        return 'unknown'
    return max(valid_scores, key=valid_scores.get)


# =========================
# Visualization Functions
# =========================
def create_word_cloud(texts, filename):
    """Generates and saves a word cloud from a list of texts."""
    print("Generating word cloud...")
    # Add any project-specific stopwords if needed
    custom_stopwords = STOPWORDS.union({'amp', 'via'}) # 'amp' often appears from '&amp;'

    # Filter out potential non-string entries before joining
    valid_texts = [text for text in texts if isinstance(text, str)]
    full_text = " ".join(valid_texts)

    if not full_text.strip():
        print("Warning: No text available for word cloud generation.")
        return

    wordcloud = WordCloud(
        width=1200, height=600,
        background_color='white',
        stopwords=custom_stopwords,
        collocations=False, # Avoid bigrams for simplicity
        max_words=200
    ).generate(full_text)

    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words in Processed Text')
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Word cloud saved to {filename}")
    except Exception as e:
        print(f"Error saving word cloud: {e}")
    plt.close()

def plot_sentiment_distribution(sentiments, filename):
    """Generates and saves a bar chart of sentiment distribution."""
    print("Generating sentiment distribution plot...")
    sentiment_counts = sentiments.value_counts().reindex(['positive', 'neutral', 'negative', 'unknown']).fillna(0)

    if sentiment_counts.sum() == 0:
        print("Warning: No sentiment data available for distribution plot.")
        return

    colors = {'positive': 'green', 'neutral': 'grey', 'negative': 'red', 'unknown': 'blue'}
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=[colors.get(s, 'black') for s in sentiment_counts.index])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=0)
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Sentiment distribution plot saved to {filename}")
    except Exception as e:
        print(f"Error saving sentiment distribution plot: {e}")
    plt.close()


# =========================
# Main Workflow
# =========================
if __name__ == "__main__":
    print("Starting Sentiment Analysis Workflow...")

    # 1. Load Data
    print(f"Loading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(
            INPUT_CSV,
            encoding=CSV_ENCODING,
            header=None,
            names=CSV_COLUMN_NAMES,
            nrows=SAMPLE_SIZE # Load only sample or all if None
            )
        print(f"Loaded {len(df)} rows.")
        if TEXT_COLUMN not in df.columns:
            raise ValueError(f"Text column '{TEXT_COLUMN}' not found in the CSV.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV}")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

    # 2. Preprocess Text
    print("Initializing text processor...")
    processor = SmartTextProcessor()
    print("Preprocessing text data...")
    # Ensure the text column is string type, fill NaNs with empty strings
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).fillna('')
    tqdm.pandas(desc="Preprocessing Text")
    df['processed_text'] = df[TEXT_COLUMN].progress_apply(processor.process)

    # Display some processed examples
    print("\nSample processed text:")
    print(df[[TEXT_COLUMN, 'processed_text']].head())

    # 3. Analyze Sentiment
    analyzer = initialize_analyzer()
    # Ensure the processed text column exists and is suitable for the analyzer
    texts_to_analyze = df['processed_text'].tolist()

    sentiment_scores = analyze_sentiment_batch(texts_to_analyze, analyzer)

    # Add scores and dominant sentiment to DataFrame
    df['sentiment_scores'] = sentiment_scores
    # Extract individual scores for easier analysis if needed
    df['neg_score'] = df['sentiment_scores'].apply(lambda x: x.get('negative') if x else None)
    df['neu_score'] = df['sentiment_scores'].apply(lambda x: x.get('neutral') if x else None)
    df['pos_score'] = df['sentiment_scores'].apply(lambda x: x.get('positive') if x else None)
    df['sentiment'] = df['sentiment_scores'].apply(get_dominant_sentiment)


    # 4. Visualization
    create_word_cloud(df['processed_text'].dropna(), WORDCLOUD_FILE)
    plot_sentiment_distribution(df['sentiment'], SENTIMENT_DIST_FILE)

    # 5. Save Results
    print(f"Saving results to {OUTPUT_CSV}...")
    try:
        # Select relevant columns to save
        output_df = df[[TEXT_COLUMN, 'processed_text', 'sentiment', 'neg_score', 'neu_score', 'pos_score']]
        output_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    print("\nAnalysis complete.")
    print(f"Output files: {OUTPUT_CSV}, {WORDCLOUD_FILE}, {SENTIMENT_DIST_FILE}")
    print("\nFinal sentiment distribution:")
    print(df['sentiment'].value_counts())