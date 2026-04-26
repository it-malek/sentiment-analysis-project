"""
Handles text preprocessing using the SmartTextProcessor class and NLTK setup.
"""
import os
import re
from typing import Set

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet


def setup_nltk(config) -> Set[str]:
    """Check for and download required NLTK packages, then return English stopwords.

    Creates the NLTK data directory specified by ``config.NLTK_DATA_DIR`` if it
    does not already exist, appends it to the NLTK search path, and downloads
    any missing packages listed in ``config.REQUIRED_NLTK_PACKAGES``.

    Args:
        config: Configuration object exposing ``NLTK_DATA_DIR`` (str or path)
            and ``REQUIRED_NLTK_PACKAGES`` (list of str package names).

    Returns:
        The set of English stopword strings loaded from the NLTK corpus.
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
                if package == "punkt":
                    nltk.data.find(f"tokenizers/{package}.zip")
                elif package == "averaged_perceptron_tagger":
                    nltk.data.find(f"taggers/{package}.zip")
                elif package == "wordnet":
                    nltk.data.find(f"corpora/{package}.zip")
                elif package == "stopwords":
                    nltk.data.find(f"corpora/{package}.zip")
                else:
                    nltk.data.find(package)
            except LookupError:
                all_packages_found = False
                print(f" - Package '{package}' not found. Downloading to: {nltk_data_dir}")
                nltk.download(package, download_dir=nltk_data_dir, quiet=False)

        if all_packages_found:
            print("All required NLTK packages found.")

        return set(stopwords.words("english"))

    except ImportError:
        print("Error: NLTK library not installed. Please install it: pip install nltk")
        exit()
    except Exception as e:
        print(f"Error during NLTK setup: {e}")
        exit()


class SmartTextProcessor:
    """Normalize tweet-style text while preserving semantically important tokens.

    Applies URL/mention stripping, POS-guided case preservation for proper
    nouns and hashtags, elongation reduction, stopword removal, and basic
    character cleaning. Designed to retain signal that plain lowercasing
    would discard.
    """

    def __init__(self, stop_words_set: Set[str]) -> None:
        """Compile regex patterns and store the stopword set.

        Args:
            stop_words_set: Set of lowercase stopword strings used to filter
                tokens after normalization.
        """
        self.url_pattern = re.compile(r"http\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.non_alpha_pattern = re.compile(r"[^a-z0-9\s.!?#]")
        self.elongation_pattern = re.compile(r"(.)\1{2,}")
        self.whitespace_pattern = re.compile(r"\s+")
        self.stop_words = stop_words_set

    def _reduce_elongation(self, word: str) -> str:
        return self.elongation_pattern.sub(r"\1\1", word)

    def process(self, text: str) -> str:
        """Normalize a single text string for NLP downstream tasks.

        Processing steps:
        1. Strip URLs and @mentions.
        2. Tokenize and POS-tag to identify proper nouns.
        3. Preserve proper nouns (NNP/NNPS), all-caps words, and hashtags
           with minimal modification.
        4. For all other tokens: lowercase, reduce character elongation,
           strip non-alphanumeric characters (except ``.!?#``).
        5. Remove stopwords and tokens shorter than 3 characters.

        Args:
            text: Raw input string. Non-string values return an empty string.

        Returns:
            A cleaned, space-joined string of normalized tokens.
        """
        if not isinstance(text, str):
            return ""

        text = self.url_pattern.sub("", text)
        text = self.mention_pattern.sub("", text)

        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
        except Exception:
            pos_tags = [(word, "UNKNOWN") for word in text.split()]

        processed_words = []
        for word, tag in pos_tags:
            if tag in ("NNP", "NNPS") or word.isupper() or word.startswith("#"):
                word = self.whitespace_pattern.sub(" ", word).strip()
                if len(word) > 1:
                    processed_words.append(word)
                continue

            word = word.lower()
            word = self._reduce_elongation(word)
            word = self.non_alpha_pattern.sub("", word)

            if word not in self.stop_words and len(word) > 2:
                if re.search(r"[a-z0-9]", word):
                    processed_words.append(word)

        processed_text = " ".join(processed_words)
        return self.whitespace_pattern.sub(" ", processed_text).strip()
