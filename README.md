# Twitter Sentiment Analysis (Adapted for Static Dataset)

**Author:** Malek Elaghel  
**Date:** May 3, 2025  
**Contact:** [malekelaghel@gmail.com](mailto:malekelaghel@gmail.com)


## Introduction & Motivation

This project performs sentiment analysis on text data, aiming to understand public opinion expressed in short texts, originally focusing on Twitter data.

### Project Origins
This work began as a summer research project during my sophomore year. The initial goal was to leverage the Twitter API v2 to fetch live tweets related to specific, timely keywords (e.g., "COVID", "lockdown", "vaccine" during 2020). The focus was on data acquisition, basic text processing (using NLTK for tokenization, stemming, stopword removal), and exploratory analysis, primarily through generating word clouds for different keywords and time periods. Early versions successfully demonstrated fetching data and visualizing word frequencies.

### Transition to Static Data
Due to the evolution of Twitter API access policies, obtaining large volumes of live tweets without significant cost is no longer feasible post-graduation/outside an academic context. Consequently, this project has been **adapted** to demonstrate the *intended full analysis pipeline* using a publicly available static dataset: **Sentiment140**.

### Current State & Purpose
This repository showcases a complete sentiment analysis workflow applied to the Sentiment140 dataset. While the data source has changed, the project fulfills the original analytical goals by demonstrating:
* Sophisticated text preprocessing tailored for noisy text.
* Advanced sentiment analysis using a state-of-the-art Transformer model.
* Analysis of sentiment trends over time.
* Extraction and visualization of popular hashtags.
* Simulated keyword/topic-based analysis on subsets of the data.

The primary purpose now is to serve as a portfolio piece, highlighting skills in Python programming, NLP techniques, data visualization, and software engineering best practices (modular code, configuration management).

## Features & Analyses

* **Data Loading & Preparation:** Loads data from CSV, handles encoding, parses dates with error handling.
* **Advanced Text Preprocessing:** Utilizes a custom `SmartTextProcessor` class with NLTK for POS tagging (to preserve context like proper nouns/hashtags), handles URL/mention removal, normalizes elongated words, and cleans irrelevant characters.
* **Transformer-based Sentiment Analysis:** Employs the `cardiffnlp/twitter-roberta-base-sentiment-latest` model via the Hugging Face `transformers` library for nuanced sentiment classification (positive, neutral, negative). Uses batch processing for efficiency.
* **Overall Visualizations:**
    * Sentiment Distribution Bar Chart (Overall)
    * Word Cloud (Overall)
    * Sentiment Trend Over Time Line Plot (Weekly frequency by default)
    * Top Hashtags Bar Chart
* **Keyword Subset Analysis:** Filters the dataset for user-defined keywords (see `config.py`) and generates separate word clouds and sentiment distribution plots for each keyword subset, simulating topic-specific analysis.

## Technology Stack

* **Python 3**
* **Pandas:** Data manipulation and loading.
* **NLTK:** Text preprocessing (tokenization, POS tagging, stopwords).
* **Transformers (Hugging Face):** Sentiment analysis model loading and inference.
* **PyTorch (or TensorFlow):** Backend for the Transformers library.
* **Matplotlib:** Generating plots.
* **WordCloud:** Generating word cloud visualizations.
* **Tqdm:** Progress bars for long processes.

## File Structure
```
sentiment-analysis-project/
│
├── main.py                 # Main execution script
├── pipeline.py             # SentimentAnalysisPipeline class (orchestrator)
├── config.py               # Configuration variables (paths, model, keywords)
├── data_loader.py          # Data loading functions
├── text_processor.py       # SmartTextProcessor class & NLTK setup
├── sentiment_analyzer.py   # Transformer model functions
├── visualizer.py           # Plotting functions
│
├── requirements.txt        # Project dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
│
├── outputs/                # Generated outputs (plots, CSV)
│   ├── keyword_analysis/   # Keyword-specific plots
│
└── nltk_data/              # NLTK data (downloaded automatically)
```


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/it-malek/sentiment-analysis-project.git
    cd sentiment-analysis-project
    ```
2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the Dataset:**
    * This project uses the Sentiment140 dataset. Due to its size, it's **not included** in the repository.
    * Download it from Kaggle: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
    * You typically need the file named `training.1600000.processed.noemoticon.csv`.
    * **Rename** this file to `sentiment140.csv`.
    * Place `sentiment140.csv` in the **root directory** of the cloned project.
5.  **NLTK Data:**
    * The necessary NLTK data packages (defined in `config.py`) will be automatically checked and downloaded to the `nltk_data/` subdirectory on the first run if they are not found. Ensure you have an internet connection for this initial setup.

## How to Run

1.  **Configure (Optional):**
    * Open `config.py` to adjust settings like:
        * `SAMPLE_SIZE`: Set to `None` to process the full dataset (warning: can take a very long time and require significant RAM/GPU memory!), or keep it as a smaller number (e.g., 1000, 5000) for quicker testing.
        * `KEYWORDS_TO_ANALYZE`: Modify the list of keywords for subset analysis.
        * `ANALYSIS_BATCH_SIZE`: Adjust based on your GPU memory if using a GPU.
        * Output filenames and directories.
2.  **Execute the main script:**
    ```bash
    python main.py
    ```
3.  **Outputs:**
    * The script will print progress updates to the console.
    * All generated outputs (plots and the final results CSV) will be saved in the `outputs/` directory.
    * Keyword-specific plots will be inside `outputs/keyword_analysis/`.

## Example Outputs

**(Example: Overall Sentiment Distribution)**
![Overall Sentiment Distribution](https://drive.usercontent.google.com/download?id=16oY4RIZyymzEP2cI_z1NG3JReVazj_9X)

**(Example: Word Cloud for "love")**
![Word Cloud for "love"](https://drive.usercontent.google.com/download?id=1SCUg2hC3llu-FsKU9wMKu7f8brEKIqjP)


## Limitations & Future Work

* **Static Dataset:** The analysis is based on the Sentiment140 dataset (from ~2009), which may not reflect current language use or topics. The original goal of analyzing live data would provide more timely insights.
* **Sample Size:** By default, the script runs on a sample (`SAMPLE_SIZE` in `config.py`) for performance reasons. Running on the full dataset requires significant resources.
* **Keyword Analysis:** The keyword analysis is a simple substring match. More advanced topic modeling (e.g., LDA) could uncover latent themes more robustly.
* **Error Analysis:** A deeper dive into misclassified sentiments could help improve the pipeline.

---

**GitHub Repository:** [https://github.com/it-malek/sentiment-analysis-project](https://github.com/it-malek/sentiment-analysis-project)
