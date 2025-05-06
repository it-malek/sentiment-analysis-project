"""
Configuration settings for the Sentiment Analysis Project.
"""

import os
import torch

# --- File Paths and Basic Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Project root directory
INPUT_CSV = os.path.join(BASE_DIR, 'sentiment140.csv') # Assumes data is in root
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
KEYWORD_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'keyword_analysis')
NLTK_DATA_DIR = os.path.join(BASE_DIR, 'nltk_data')

# --- Dataset Config ---
CSV_ENCODING = 'latin-1'
CSV_COLUMN_NAMES = ['target', 'ids', 'date', 'flag', 'user', 'text']
TEXT_COLUMN = 'text'
DATE_COLUMN = 'date'
SAMPLE_SIZE = 2500 # Set to None to process the entire dataset (can be slow!)
RANDOM_STATE = 0 # For reproducible sampling if SAMPLE_SIZE is used

# --- Analysis Config ---
# Keywords for subset analysis (relevant to general text like Sentiment140)
KEYWORDS_TO_ANALYZE = ['love', 'hate', 'good', 'bad', 'happy', 'sad', 'work', 'school', 'music', 'food']
HASHTAG_TOP_N = 10 # Number of top hashtags to plot
TIME_ANALYSIS_FREQ = 'W' # Resample frequency: 'D'=Day, 'W'=Week, 'M'=Month

# --- Model Config ---
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Specify device (GPU if available, else CPU)
DEVICE = 0 if torch.cuda.is_available() else -1
# Batch size for sentiment analysis (adjust based on GPU memory)
ANALYSIS_BATCH_SIZE = 32
# Max sequence length for tokenizer (RoBERTa base models typically use 512)
TOKENIZER_MAX_LEN = 512

# --- Output Filenames ---
# These will be saved inside OUTPUT_DIR
OUTPUT_CSV = 'sentiment_analysis_results.csv'
WORDCLOUD_OVERALL_FILE = 'wordcloud_overall.png'
SENTIMENT_DIST_OVERALL_FILE = 'sentiment_distribution_overall.png'
SENTIMENT_TREND_FILE = 'sentiment_trend.png'
HASHTAG_PLOT_FILE = 'top_hashtags.png'

# Keyword plot filename templates (will be saved in KEYWORD_OUTPUT_DIR)
KEYWORD_WC_TEMPLATE = 'wordcloud_{keyword}.png'
KEYWORD_DIST_TEMPLATE = 'sentiment_{keyword}.png'

# --- NLTK Setup ---
# List of required NLTK packages
REQUIRED_NLTK_PACKAGES = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']