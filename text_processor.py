"""
Handles text preprocessing using the SmartTextProcessor class and NLTK setup.
"""
import os
import re
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet, stopwords

# --- NLTK Setup ---
def setup_nltk(config):
    """
    Checks for and downloads required NLTK data packages if necessary.
    Sets up the NLTK data path.
    Returns:
        set: The set of English stopwords.
    """
    try:
        nltk_data_dir = config.NLTK_DATA_DIR
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
            print(f"Created NLTK data directory: {nltk_data_dir}")
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)
            print(f"Added {nltk_data_dir} to NLTK data path.")

        print("Checking NLTK packages...")
        all_packages_found = True
        for package in config.REQUIRED_NLTK_PACKAGES:
            try:
                # Adjust find path based on package type
                if package == 'punkt':
                    nltk.data.find(f'tokenizers/{package}.zip')
                elif package == 'averaged_perceptron_tagger':
                    nltk.data.find(f'taggers/{package}.zip')
                elif package == 'wordnet':
                    nltk.data.find(f'corpora/{package}.zip')
                elif package == 'stopwords':
                    nltk.data.find(f'corpora/{package}.zip')
                else:
                    nltk.data.find(package) # Fallback for other types
                # print(f" - Found: {package}")
            except LookupError:
                all_packages_found = False
                print(f" - Package '{package}' not found. Downloading to: {nltk_data_dir}")
                nltk.download(package, download_dir=nltk_data_dir, quiet=False) # Set quiet=False for visibility

        if all_packages_found:
            print("All required NLTK packages found.")

        return set(stopwords.words('english'))

    except ImportError:
        print("Error: NLTK library not installed. Please install it: pip install nltk")
        exit()
    except Exception as e:
        print(f"Error during NLTK setup: {e}")
        exit()

# --- Text Processor Class ---
class SmartTextProcessor:
    """
    Processes text by normalizing common artifacts like elongations,
    while attempting to preserve context using POS tagging.
    """
    def __init__(self, stop_words_set):
        self.url_pattern = re.compile(r"http\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        # Keep hashtags for model, remove non-alpha but keep basic punctuation
        self.non_alpha_pattern = re.compile(r"[^a-z0-9\s.!?#]") # Keep # . ! ?
        self.elongation_pattern = re.compile(r'(.)\1{2,}')
        self.whitespace_pattern = re.compile(r'\s+')
        self.stop_words = stop_words_set

    def _reduce_elongation(self, word):
        return self.elongation_pattern.sub(r'\1\1', word)

    def process(self, text):
        if not isinstance(text, str): return ""
        # Basic cleaning first
        text = self.url_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)

        # Tokenize preserving case for POS tagging
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
        except Exception as e:
            # Fallback if tokenization/tagging fails
            pos_tags = [(word, 'UNKNOWN') for word in text.split()]

        processed_words = []
        for word, tag in pos_tags:
            # Preserve proper nouns, uppercase words, hashtags
            if tag in ['NNP', 'NNPS'] or word.isupper() or word.startswith('#'):
                 word = self.whitespace_pattern.sub(' ', word).strip()
                 if len(word) > 1: processed_words.append(word)
                 continue

            # Process others: lowercase, reduce elongation, clean non-alpha
            word = word.lower()
            word = self._reduce_elongation(word)
            word = self.non_alpha_pattern.sub('', word) # Clean remaining chars

            # Remove stopwords and short words *after* cleaning
            if word not in self.stop_words and len(word) > 2:
                # Ensure it's not just punctuation remnants
                if re.search(r'[a-z0-9]', word):
                     processed_words.append(word)

        processed_text = " ".join(processed_words)
        return self.whitespace_pattern.sub(' ', processed_text).strip()