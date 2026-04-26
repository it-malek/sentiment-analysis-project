# -*- coding: utf-8 -*-
"""
Functions for initializing and running the Hugging Face sentiment analysis pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Pipeline, pipeline


def initialize_analyzer(config) -> Pipeline:
    """Initialize the Hugging Face sentiment analysis pipeline.

    Args:
        config: Configuration object exposing ``SENTIMENT_MODEL`` (str, the
            Hugging Face model identifier) and ``DEVICE`` (int, 0 for GPU
            or -1 for CPU).

    Returns:
        A Hugging Face ``Pipeline`` configured for sentiment analysis with
        ``return_all_scores=True``.
    """
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
            return_all_scores=True,
        )
        print("Sentiment model loaded successfully.")
        return sentiment_analyzer
    except Exception as e:
        print(f"Error loading sentiment model '{model_name}': {e}")
        print("Check model name, internet connection, and installed libraries (transformers, torch/tensorflow).")
        exit()


def analyze_sentiment_batch(
    texts: Iterable[Any],
    analyzer: Pipeline,
    config,
) -> List[Dict[str, Optional[float]]]:
    """Run batch sentiment inference over a collection of texts.

    Skips NaN values and empty strings, processes the remaining texts in
    batches, and returns one score dictionary per input element.

    Args:
        texts: Iterable of raw text values. May include NaN entries, which
            are silently dropped before inference.
        analyzer: An initialized Hugging Face ``Pipeline`` with
            ``return_all_scores=True``.
        config: Configuration object exposing ``ANALYSIS_BATCH_SIZE`` (int)
            and ``TOKENIZER_MAX_LEN`` (int).

    Returns:
        A list of dicts mapping sentiment label strings (``"positive"``,
        ``"neutral"``, ``"negative"``) to float confidence scores, one entry
        per input text. Entries for empty or failed texts map every label to
        ``None``.
    """
    results = []
    batch_size = config.ANALYSIS_BATCH_SIZE
    max_len = config.TOKENIZER_MAX_LEN

    print(f"Analyzing sentiment in batches of {batch_size}...")
    texts_list = [str(text) for text in texts if pd.notna(text)]

    for i in tqdm(range(0, len(texts_list), batch_size), desc="Sentiment Analysis"):
        batch = texts_list[i:i + batch_size]
        valid_batch = [text for text in batch if len(text.strip()) > 0]

        if not valid_batch:
            results.extend([None] * len(batch))
            continue

        try:
            with torch.no_grad():
                batch_results = analyzer(valid_batch, truncation=True, max_length=max_len, top_k=None)
            result_map = {text: res for text, res in zip(valid_batch, batch_results)}
            results.extend([result_map.get(text) for text in batch])
        except Exception as e:
            print(f"\nError processing sentiment batch starting at index {i}: {e}")
            print(f"Problematic batch content (first few items): {batch[:5]}")
            results.extend([None] * len(batch))

    processed_results: List[Dict[str, Optional[float]]] = []
    for res_list in results:
        if isinstance(res_list, list) and all(
            isinstance(item, dict) and "label" in item and "score" in item
            for item in res_list
        ):
            score_dict = {item["label"].lower(): item["score"] for item in res_list}
            processed_results.append(score_dict)
        else:
            processed_results.append({"negative": None, "neutral": None, "positive": None})
    return processed_results


def get_dominant_sentiment(score_dict: Dict[str, Optional[float]]) -> str:
    """Return the highest-scoring sentiment label from a score dictionary.

    Handles label variations across different model architectures by mapping
    them to one of three canonical labels: ``"positive"``, ``"neutral"``, or
    ``"negative"``.

    Args:
        score_dict: Dict mapping raw model label strings to float confidence
            scores. Values may be ``None`` for failed predictions.

    Returns:
        The dominant sentiment string (``"positive"``, ``"neutral"``, or
        ``"negative"``), or ``"unknown"`` when ``score_dict`` is invalid or
        all scores are ``None``.
    """
    label_map = {
        "negative": "negative", "neutral": "neutral", "positive": "positive",
        "neg": "negative", "neu": "neutral", "pos": "positive",
        "label_0": "negative", "label_1": "neutral", "label_2": "positive",
    }

    if not isinstance(score_dict, dict) or all(v is None for v in score_dict.values()):
        return "unknown"

    relevant_scores: Dict[str, float] = {}
    for model_label, score in score_dict.items():
        standard_label = label_map.get(model_label.lower())
        if standard_label and score is not None:
            relevant_scores[standard_label] = max(score, relevant_scores.get(standard_label, -1.0))

    if not relevant_scores or all(v == -1.0 for v in relevant_scores.values()):
        return "unknown"

    return max(relevant_scores, key=relevant_scores.get)
