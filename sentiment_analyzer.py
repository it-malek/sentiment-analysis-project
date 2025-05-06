# -*- coding: utf-8 -*-
"""
Functions for initializing and running the Hugging Face sentiment analysis pipeline.
"""
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import pipeline, AutoTokenizer, logging as hf_logging

# Suppress verbose logging from Hugging Face load messages if desired
# hf_logging.set_verbosity_error()

def initialize_analyzer(config):
    """Initializes the Hugging Face sentiment analysis pipeline."""
    model_name = config.SENTIMENT_MODEL
    device = config.DEVICE
    print(f"Loading sentiment analysis model: {model_name}...")
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True # Get scores for all labels
        )
        print("Sentiment model loaded successfully.")
        return sentiment_analyzer
    except Exception as e:
        print(f"Error loading sentiment model '{model_name}': {e}")
        print("Check model name, internet connection, and installed libraries (transformers, torch/tensorflow).")
        exit()

def analyze_sentiment_batch(texts, analyzer, config):
    """Analyzes sentiment for a list of texts using batch processing."""
    results = []
    batch_size = config.ANALYSIS_BATCH_SIZE
    max_len = config.TOKENIZER_MAX_LEN

    print(f"Analyzing sentiment in batches of {batch_size}...")
    # Ensure texts is a list of strings, handle potential NaNs
    texts_list = [str(text) for text in texts if pd.notna(text)]

    for i in tqdm(range(0, len(texts_list), batch_size), desc="Sentiment Analysis"):
        batch = texts_list[i:i + batch_size]
        valid_batch = [text for text in batch if len(text.strip()) > 0]

        if not valid_batch:
            results.extend([None] * len(batch))
            continue

        try:
            # Disable gradient calculation for efficiency during inference
            with torch.no_grad():
                 batch_results = analyzer(valid_batch, truncation=True, max_length=max_len, top_k=None)
            # Map results back to the original batch order (including empty strings)
            result_map = {text: res for text, res in zip(valid_batch, batch_results)}
            results.extend([result_map.get(text) for text in batch])
        except Exception as e:
            print(f"\nError processing sentiment batch starting at index {i}: {e}")
            print(f"Problematic batch content (first few items): {batch[:5]}")
            results.extend([None] * len(batch)) # Add None for failed items

    # Post-process results into a consistent dictionary format
    processed_results = []
    for res_list in results:
        # Check if result is a list of dicts with 'label' and 'score'
        if isinstance(res_list, list) and all(isinstance(item, dict) and 'label' in item and 'score' in item for item in res_list):
             score_dict = {item['label'].lower(): item['score'] for item in res_list}
             processed_results.append(score_dict)
        else: # Handle None or unexpected format from pipeline/error
             processed_results.append({'negative': None, 'neutral': None, 'positive': None})
    return processed_results


def get_dominant_sentiment(score_dict):
    """Determines the dominant sentiment label from a score dictionary."""
    # Map model output labels (potentially different across models) to standard labels
    label_map = {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive',
                 'neg': 'negative', 'neu': 'neutral', 'pos': 'positive', # Common variations
                 'label_0': 'negative', 'label_1': 'neutral', 'label_2': 'positive'} # Example if model returns labels like this

    if not isinstance(score_dict, dict) or all(v is None for v in score_dict.values()):
        return 'unknown'

    # Convert keys to standard labels using map, handle missing keys gracefully
    relevant_scores = {}
    for model_label, score in score_dict.items():
        standard_label = label_map.get(model_label.lower())
        if standard_label and score is not None:
            # Take the highest score if multiple model labels map to the same standard label
            relevant_scores[standard_label] = max(score, relevant_scores.get(standard_label, -1.0))

    if not relevant_scores or all(v == -1.0 for v in relevant_scores.values()):
        return 'unknown'

    # Find the standard label with the highest score
    return max(relevant_scores, key=relevant_scores.get)