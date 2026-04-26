# -*- coding: utf-8 -*-
"""
Functions for generating and saving visualizations (plots and word clouds).
Includes improved aesthetics and integer ticks for frequency axes.
"""
import os
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

plt.style.use("seaborn-v0_8-whitegrid")


def create_output_dir(config) -> None:
    """Create output directories specified in config if they do not exist.

    Args:
        config: Configuration object exposing ``OUTPUT_DIR`` and
            ``KEYWORD_OUTPUT_DIR`` (str or path).
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.KEYWORD_OUTPUT_DIR, exist_ok=True)


def _save_plot(figure, filename, config, is_keyword_subset: bool) -> None:
    """Save a matplotlib figure to the appropriate output directory."""
    if is_keyword_subset:
        filepath = os.path.join(config.KEYWORD_OUTPUT_DIR, filename)
    else:
        filepath = os.path.join(config.OUTPUT_DIR, filename)
    try:
        figure.savefig(filepath, bbox_inches="tight")
        print(f"Plot saved to {filepath}")
    except Exception as e:
        print(f"Error saving plot '{filepath}': {e}")
    finally:
        plt.close(figure)


def create_word_cloud(
    texts,
    filename: str,
    title: str,
    config,
    is_keyword_subset: bool = False,
) -> None:
    """Generate and save a word cloud from a collection of texts.

    Hashtags are extracted and added to the stopword list so they do not
    dominate the visualization; the remaining token frequencies drive the
    layout.

    Args:
        texts: Iterable of raw text strings to visualize.
        filename: Output filename (saved inside the appropriate output dir).
        title: Plot title displayed above the word cloud.
        config: Configuration object exposing ``OUTPUT_DIR`` and
            ``KEYWORD_OUTPUT_DIR``.
        is_keyword_subset: When ``True``, saves into the keyword-analysis
            subdirectory instead of the main output directory.
    """
    print(f"Generating word cloud: {title}...")

    custom_stopwords = STOPWORDS.union({"amp", "via", "rt", "u", "im", "co", "http", "https"})
    all_hashtags: set = set()
    valid_texts_for_wc = []
    for text in texts:
        if isinstance(text, str):
            clean_text = re.sub(r"#\w+", "", text)
            valid_texts_for_wc.append(clean_text)
            all_hashtags.update(tag.strip("#").lower() for tag in re.findall(r"#\w+", text))
    custom_stopwords.update(all_hashtags)

    full_text = " ".join(valid_texts_for_wc)
    if not full_text.strip():
        print(f"Warning: No text available for word cloud '{title}'.")
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    try:
        wordcloud = WordCloud(
            width=1200, height=600, background_color="white",
            stopwords=custom_stopwords, collocations=False, max_words=150,
        ).generate(full_text)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=18)
        _save_plot(fig, filename, config, is_keyword_subset)
    except Exception as e:
        print(f"Error generating/saving word cloud '{title}': {e}")
        plt.close(fig)


def plot_sentiment_distribution(
    sentiments,
    filename: str,
    title: str,
    config,
    is_keyword_subset: bool = False,
) -> None:
    """Generate and save a bar chart of sentiment label counts.

    Args:
        sentiments: pandas Series of sentiment label strings.
        filename: Output filename.
        title: Plot title.
        config: Configuration object exposing output directory paths.
        is_keyword_subset: When ``True``, saves into the keyword-analysis
            subdirectory.
    """
    print(f"Generating sentiment distribution plot: {title}...")
    if sentiments.empty:
        print(f"Warning: No sentiment data for distribution plot '{title}'.")
        return

    sentiment_counts = (
        sentiments.value_counts()
        .reindex(["positive", "neutral", "negative", "unknown"])
        .fillna(0)
    )
    if sentiment_counts.sum() == 0:
        print(f"Warning: No valid sentiment counts for distribution plot '{title}'.")
        return

    colors = {
        "positive": "mediumseagreen",
        "neutral": "lightgray",
        "negative": "lightcoral",
        "unknown": "lightskyblue",
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    sentiment_counts.plot(
        kind="bar",
        color=[colors.get(s, "black") for s in sentiment_counts.index],
        ax=ax,
        zorder=3,
    )
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Number of Records", fontsize=12)
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    plt.tight_layout(pad=1.5)
    _save_plot(fig, filename, config, is_keyword_subset)


def plot_sentiment_over_time(df, config) -> None:
    """Plot sentiment label frequencies over time as a line chart.

    Resamples the dataframe at the frequency specified by
    ``config.TIME_ANALYSIS_FREQ`` and renders one line per sentiment label.
    Skips gracefully when the date column is missing or fully null.

    Args:
        df: pandas DataFrame containing at minimum a parsed datetime column
            (``config.DATE_COLUMN``) and a ``"sentiment"`` string column.
        config: Configuration object exposing ``DATE_COLUMN``,
            ``TIME_ANALYSIS_FREQ``, ``SENTIMENT_TREND_FILE``,
            ``OUTPUT_DIR``, and ``KEYWORD_OUTPUT_DIR``.
    """
    date_col = config.DATE_COLUMN
    sentiment_col = "sentiment"
    filename = config.SENTIMENT_TREND_FILE
    freq = config.TIME_ANALYSIS_FREQ

    print(f"Generating sentiment over time plot ({filename})...")
    if date_col not in df.columns or df[date_col].isnull().all():
        print(f"Warning: Valid date column ('{date_col}') not found or empty. Skipping time plot.")
        return
    if sentiment_col not in df.columns:
        print(f"Warning: Sentiment column ('{sentiment_col}') not found. Skipping time plot.")
        return

    df_time = df.dropna(subset=[date_col, sentiment_col]).copy()
    df_time[date_col] = pd.to_datetime(df_time[date_col], errors="coerce")
    df_time = df_time.dropna(subset=[date_col])
    if df_time.empty:
        print("Warning: No valid date data for time plot.")
        return
    df_time = df_time.sort_values(by=date_col)
    sentiment_over_time = (
        df_time.groupby([pd.Grouper(key=date_col, freq=freq), sentiment_col])
        .size()
        .unstack(fill_value=0)
    )
    if sentiment_over_time.empty:
        print(f"Warning: No data after resampling by '{freq}' for sentiment trend plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    plot_colors = {
        "positive": "mediumseagreen",
        "neutral": "darkgrey",
        "negative": "lightcoral",
        "unknown": "lightskyblue",
    }
    plot_order = [col for col in ["positive", "neutral", "negative", "unknown"] if col in sentiment_over_time.columns]
    sentiment_over_time[plot_order].plot(
        kind="line", ax=ax,
        color=[plot_colors[col] for col in plot_order],
        linewidth=1.5, marker="o", markersize=3, linestyle="-",
    )
    ax.set_title(f"Sentiment Trends Over Time (Frequency: {freq})", fontsize=18, pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Records", fontsize=12)
    ax.legend(title="Sentiment", title_fontsize="11", fontsize="10")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    num_periods = len(sentiment_over_time)
    interval = max(1, num_periods // 10)
    if freq == "D":
        locator = mdates.DayLocator(interval=interval)
    elif freq == "W":
        locator = mdates.WeekdayLocator(interval=interval)
    else:
        locator = mdates.MonthLocator(interval=interval)
    ax.xaxis.set_major_locator(locator)
    fig.autofmt_xdate()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.grid(axis="both", linestyle=":", alpha=0.6)
    plt.tight_layout(pad=1.5)
    _save_plot(fig, filename, config, is_keyword_subset=False)


def analyze_hashtags(df, config) -> None:
    """Extract, rank, and plot the top N hashtags as a horizontal bar chart.

    Args:
        df: pandas DataFrame with a text column containing raw tweet strings.
        config: Configuration object exposing ``TEXT_COLUMN`` (str),
            ``HASHTAG_PLOT_FILE`` (str filename), ``HASHTAG_TOP_N`` (int),
            ``OUTPUT_DIR``, and ``KEYWORD_OUTPUT_DIR``.
    """
    text_col = config.TEXT_COLUMN
    filename = config.HASHTAG_PLOT_FILE
    top_n = config.HASHTAG_TOP_N

    print(f"Analyzing hashtags and generating plot ({filename})...")
    if text_col not in df.columns:
        print(f"Warning: Text column ('{text_col}') not found for hashtag analysis.")
        return
    hashtags = df[text_col].astype(str).str.findall(r"#(\w+)").explode().str.lower().dropna()
    if hashtags.empty:
        print("Warning: No hashtags found in the dataset.")
        return
    top_hashtags = hashtags.value_counts().head(top_n)
    if top_hashtags.empty:
        print("Warning: No hashtags to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))
    top_hashtags.sort_values().plot(kind="barh", ax=ax, color="steelblue", zorder=3)
    ax.set_title(f"Top {top_n} Hashtags", fontsize=16, pad=15)
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Hashtag", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, min_n_ticks=3))
    ax.set_xlim(left=0)
    ax.grid(axis="x", linestyle="--", alpha=0.7, zorder=0)
    plt.tight_layout(pad=1.5)
    _save_plot(fig, filename, config, is_keyword_subset=False)
