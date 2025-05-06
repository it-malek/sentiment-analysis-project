# -*- coding: utf-8 -*-
"""
Functions for generating and saving visualizations (plots and word clouds).
Includes improved aesthetics and integer ticks for frequency axes.
"""
import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker # Import ticker for integer ticks
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

# --- Apply Global Plot Style ---
# Use a style for better aesthetics (e.g., includes grid, better default colors)
# Other good options: 'seaborn-v0_8-pastel', 'ggplot', 'fivethirtyeight'
plt.style.use('seaborn-v0_8-whitegrid')
# Optionally set a default font family (Matplotlib will search for available fonts)
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica'] # Example fallback order


def create_output_dir(config):
    """Creates output directories if they don't exist."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.KEYWORD_OUTPUT_DIR, exist_ok=True)

def _save_plot(figure, filename, config, is_keyword_subset):
    """Helper function to save plots to the correct directory."""
    if is_keyword_subset:
        filepath = os.path.join(config.KEYWORD_OUTPUT_DIR, filename)
    else:
        filepath = os.path.join(config.OUTPUT_DIR, filename)
    try:
        figure.savefig(filepath, bbox_inches='tight') # Use bbox_inches='tight'
        print(f"Plot saved to {filepath}")
    except Exception as e:
        print(f"Error saving plot '{filepath}': {e}")
    finally:
        plt.close(figure) # Close the specific figure

# --- Visualization Functions ---

def create_word_cloud(texts, filename, title, config, is_keyword_subset=False):
    """Generates and saves a word cloud with adjusted title size."""
    print(f"Generating word cloud: {title}...")

    # Define stopwords (same logic as before)
    custom_stopwords = STOPWORDS.union({'amp', 'via', 'rt', 'u', 'im', 'co', 'http', 'https'})
    all_hashtags = set()
    valid_texts_for_wc = []
    for text in texts:
         if isinstance(text, str):
             clean_text = re.sub(r'#\w+', '', text)
             valid_texts_for_wc.append(clean_text)
             all_hashtags.update(tag.strip('#').lower() for tag in re.findall(r'#\w+', text))
    custom_stopwords.update(all_hashtags)

    full_text = " ".join(valid_texts_for_wc)
    if not full_text.strip():
        print(f"Warning: No text available for word cloud '{title}'.")
        return

    # Create the figure object explicitly
    fig, ax = plt.subplots(figsize=(15, 7))

    try:
        # Generate word cloud object
        wordcloud = WordCloud(
            width=1200, height=600, background_color='white',
            stopwords=custom_stopwords, collocations=False, max_words=150
        ).generate(full_text)

        # Display the word cloud on the axes
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=18) # Slightly larger title font

        # Save the plot using the helper
        _save_plot(fig, filename, config, is_keyword_subset)

    except Exception as e:
        print(f"Error generating/saving word cloud '{title}': {e}")
        plt.close(fig) # Ensure figure is closed on error


def plot_sentiment_distribution(sentiments, filename, title, config, is_keyword_subset=False):
    """Generates and saves a bar chart of sentiment distribution with integer y-axis."""
    print(f"Generating sentiment distribution plot: {title}...")
    if sentiments.empty:
        print(f"Warning: No sentiment data for distribution plot '{title}'.")
        return

    sentiment_counts = sentiments.value_counts().reindex(['positive', 'neutral', 'negative', 'unknown']).fillna(0)
    if sentiment_counts.sum() == 0:
        print(f"Warning: No valid sentiment counts for distribution plot '{title}'.")
        return

    colors = {'positive': 'mediumseagreen', 'neutral': 'lightgray', 'negative': 'lightcoral', 'unknown': 'lightskyblue'} # Softer colors

    # Create figure and axes objects
    fig, ax = plt.subplots(figsize=(9, 6)) # Adjusted size slightly

    # Plot on the specific axes object
    sentiment_counts.plot(kind='bar', color=[colors.get(s, 'black') for s in sentiment_counts.index], ax=ax, zorder=3) # zorder=3 to plot bars above grid

    # --- Customizations ---
    ax.set_title(title, fontsize=16, pad=15) # Add padding
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Number of Records', fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=11) # Customize tick labels
    ax.tick_params(axis='y', labelsize=11)

    # Force integer ticks on the Y-axis and ensure it starts at 0
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)

    # Add grid lines (often included with seaborn styles, but explicit control is good)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0) # Grid behind bars

    # Improve layout
    plt.tight_layout(pad=1.5)

    # Save using helper
    _save_plot(fig, filename, config, is_keyword_subset)


def plot_sentiment_over_time(df, config):
    """ Plots sentiment distribution over time with integer y-axis. """
    date_col = config.DATE_COLUMN
    sentiment_col = 'sentiment'
    filename = config.SENTIMENT_TREND_FILE
    freq = config.TIME_ANALYSIS_FREQ

    print(f"Generating sentiment over time plot ({filename})...")
    # --- Data Validation (same as before) ---
    if date_col not in df.columns or df[date_col].isnull().all():
        print(f"Warning: Valid date column ('{date_col}') not found or empty. Skipping time plot.")
        return
    if sentiment_col not in df.columns:
         print(f"Warning: Sentiment column ('{sentiment_col}') not found. Skipping time plot.")
         return
    df_time = df.dropna(subset=[date_col, sentiment_col]).copy()
    df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
    df_time = df_time.dropna(subset=[date_col])
    if df_time.empty:
        print("Warning: No valid date data for time plot.")
        return
    df_time = df_time.sort_values(by=date_col)
    sentiment_over_time = df_time.groupby([pd.Grouper(key=date_col, freq=freq), sentiment_col]).size().unstack(fill_value=0)
    if sentiment_over_time.empty:
        print(f"Warning: No data after resampling by '{freq}' for sentiment trend plot.")
        return
    # --- End Data Validation ---

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(16, 7)) # Slightly wider

    # Define plot colors and order
    plot_colors = {'positive': 'mediumseagreen', 'neutral': 'darkgrey', 'negative': 'lightcoral', 'unknown': 'lightskyblue'}
    plot_order = [col for col in ['positive', 'neutral', 'negative', 'unknown'] if col in sentiment_over_time.columns]

    # Plot data on the axes
    sentiment_over_time[plot_order].plot(kind='line', ax=ax, color=[plot_colors[col] for col in plot_order], linewidth=1.5, marker='o', markersize=3, linestyle='-') # Add markers

    # --- Customizations ---
    ax.set_title(f'Sentiment Trends Over Time (Frequency: {freq})', fontsize=18, pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Records', fontsize=12)
    ax.legend(title='Sentiment', title_fontsize='11', fontsize='10')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Date axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # Adjust locator interval based on data range or frequency more dynamically
    num_periods = len(sentiment_over_time)
    interval = max(1, num_periods // 10) # Aim for about 10 major ticks
    if freq == 'D': locator = mdates.DayLocator(interval=interval)
    elif freq == 'W': locator = mdates.WeekdayLocator(interval=interval)
    else: locator = mdates.MonthLocator(interval=interval)
    ax.xaxis.set_major_locator(locator)
    fig.autofmt_xdate() # Auto format dates to prevent overlap

    # Force integer ticks on the Y-axis and ensure it starts at 0
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)

    # Add subtle grid
    ax.grid(axis='both', linestyle=':', alpha=0.6)

    # Improve layout
    plt.tight_layout(pad=1.5)

    # Save using helper
    _save_plot(fig, filename, config, is_keyword_subset=False)


def analyze_hashtags(df, config):
    """ Extracts, counts, and plots top N hashtags with integer x-axis. """
    text_col = config.TEXT_COLUMN
    filename = config.HASHTAG_PLOT_FILE
    top_n = config.HASHTAG_TOP_N

    print(f"Analyzing hashtags and generating plot ({filename})...")
    # --- Data Validation (same as before) ---
    if text_col not in df.columns:
        print(f"Warning: Text column ('{text_col}') not found for hashtag analysis.")
        return
    hashtags = df[text_col].astype(str).str.findall(r'#(\w+)').explode().str.lower().dropna()
    if hashtags.empty:
        print("Warning: No hashtags found in the dataset.")
        return
    top_hashtags = hashtags.value_counts().head(top_n)
    if top_hashtags.empty:
        print("Warning: No hashtags to plot.")
        return
    # --- End Data Validation ---

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45))) # Slightly more vertical space per item

    # Plot horizontal bar chart on axes
    top_hashtags.sort_values().plot(kind='barh', ax=ax, color='steelblue', zorder=3)

    # --- Customizations ---
    ax.set_title(f'Top {top_n} Hashtags', fontsize=16, pad=15)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Hashtag', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11) # Hashtag labels

    # Force integer ticks on the X-axis and ensure it starts at 0
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, min_n_ticks=3)) # Ensure at least a few ticks
    ax.set_xlim(left=0)

    # Add grid lines for the x-axis (frequency)
    ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)

    # Improve layout
    plt.tight_layout(pad=1.5)

    # Save using helper
    _save_plot(fig, filename, config, is_keyword_subset=False)