"""
Contains the main SentimentAnalysisPipeline class orchestrating the workflow.
"""
import pandas as pd
from tqdm.auto import tqdm
import os

# Import necessary components from other modules
import data_loader
import text_processor
import sentiment_analyzer
import visualizer

# Enable tqdm progress bar for pandas apply functions
tqdm.pandas()

class SentimentAnalysisPipeline:
    """
    Orchestrates the sentiment analysis workflow including data loading,
    preprocessing, analysis, and visualization.
    """
    def __init__(self, config):
        """
        Initializes the pipeline with configuration and components.
        Args:
            config: The configuration module.
        """
        self.config = config
        self.df = None # Dataframe will be loaded here
        self.stop_words = None
        self.processor = None
        self.analyzer = None

        # Create output directories early
        visualizer.create_output_dir(self.config)

    def _setup_components(self):
        """Initializes processor and analyzer after NLTK setup."""
        self.stop_words = text_processor.setup_nltk(self.config)
        self.processor = text_processor.SmartTextProcessor(self.stop_words)
        self.analyzer = sentiment_analyzer.initialize_analyzer(self.config)

    def load_and_prep_data(self):
        """Loads data using data_loader."""
        print("\n--- Loading and Preparing Data ---")
        self.df = data_loader.load_data(self.config)
        if self.df is None:
            raise RuntimeError("Data loading failed. Please check errors above.")
        # Setup NLP components after data is loaded (in case NLTK needs download)
        self._setup_components()


    def process_text(self):
        """Applies text preprocessing to the dataframe."""
        if self.df is None or self.processor is None:
            raise RuntimeError("Dataframe or processor not initialized. Run load_and_prep_data first.")
        print("\n--- Preprocessing Text ---")
        text_col = self.config.TEXT_COLUMN
        self.df['processed_text'] = self.df[text_col].progress_apply(self.processor.process)
        print("Text preprocessing complete.")
        print("Sample processed text:")
        print(self.df[[text_col, 'processed_text']].head())


    def run_sentiment_analysis(self):
        """Runs batch sentiment analysis."""
        if self.df is None or self.analyzer is None:
             raise RuntimeError("Dataframe or analyzer not initialized.")
        print("\n--- Running Sentiment Analysis ---")
        texts_to_analyze = self.df['processed_text'].tolist()
        sentiment_scores = sentiment_analyzer.analyze_sentiment_batch(texts_to_analyze, self.analyzer, self.config)

        # Add results back to DataFrame
        self.df['sentiment_scores'] = sentiment_scores
        self.df['neg_score'] = self.df['sentiment_scores'].apply(lambda x: x.get('negative') if isinstance(x, dict) else None)
        self.df['neu_score'] = self.df['sentiment_scores'].apply(lambda x: x.get('neutral') if isinstance(x, dict) else None)
        self.df['pos_score'] = self.df['sentiment_scores'].apply(lambda x: x.get('positive') if isinstance(x, dict) else None)
        self.df['sentiment'] = self.df['sentiment_scores'].apply(sentiment_analyzer.get_dominant_sentiment)
        print("Sentiment analysis complete.")
        print("Overall sentiment distribution:")
        print(self.df['sentiment'].value_counts())


    def generate_visualizations(self):
        """Generates and saves all standard visualizations."""
        if self.df is None: raise RuntimeError("Dataframe not available for visualization.")
        print("\n--- Generating Overall Visualizations ---")
        # Overall Word Cloud
        visualizer.create_word_cloud(
            self.df['processed_text'].dropna(),
            self.config.WORDCLOUD_OVERALL_FILE,
            'Overall Word Cloud',
            self.config
        )
        # Overall Sentiment Distribution
        visualizer.plot_sentiment_distribution(
            self.df['sentiment'],
            self.config.SENTIMENT_DIST_OVERALL_FILE,
            'Overall Sentiment Distribution',
            self.config
        )
        # Sentiment Trend Over Time
        visualizer.plot_sentiment_over_time(self.df, self.config)
        # Hashtag Analysis
        visualizer.analyze_hashtags(self.df, self.config)
        print("Overall visualizations generated.")


    def analyze_keyword_subsets(self):
        """Analyzes and visualizes subsets based on keywords."""
        if self.df is None: raise RuntimeError("Dataframe not available for keyword analysis.")
        print("\n--- Analyzing Keyword Subsets ---")
        text_col = self.config.TEXT_COLUMN
        keywords = self.config.KEYWORDS_TO_ANALYZE

        for keyword in keywords:
            print(f"\nAnalyzing subset for keyword: '{keyword}'")
            # Filter dataframe (case-insensitive search in original text)
            keyword_df = self.df[self.df[text_col].str.contains(keyword, case=False, na=False)]

            if not keyword_df.empty:
                subset_size = len(keyword_df)
                print(f"Found {subset_size} records containing '{keyword}'.")

                # Define output filenames using templates from config
                kw_wc_file = self.config.KEYWORD_WC_TEMPLATE.format(keyword=keyword.replace(" ", "_"))
                kw_dist_file = self.config.KEYWORD_DIST_TEMPLATE.format(keyword=keyword.replace(" ", "_"))

                # Generate visualizations for the subset, indicating it's a keyword plot
                visualizer.create_word_cloud(
                    keyword_df['processed_text'].dropna(),
                    kw_wc_file,
                    f"Word Cloud for '{keyword}' ({subset_size} records)",
                    self.config,
                    is_keyword_subset=True # Flag for saving location
                )
                visualizer.plot_sentiment_distribution(
                    keyword_df['sentiment'],
                    kw_dist_file,
                    f"Sentiment for '{keyword}' ({subset_size} records)",
                    self.config,
                    is_keyword_subset=True # Flag for saving location
                )
            else:
                print(f"No records found containing '{keyword}'.")
        print("\n--- Keyword Subset Analysis Complete ---")


    def save_results(self):
        """Saves the final dataframe with analysis results to a CSV file."""
        if self.df is None: raise RuntimeError("Dataframe not available to save.")
        print("\n--- Saving Results ---")
        output_path = os.path.join(self.config.OUTPUT_DIR, self.config.OUTPUT_CSV)
        try:
            # Select relevant columns (ensure date column exists if included)
            output_cols = [self.config.TEXT_COLUMN, 'processed_text']
            if self.config.DATE_COLUMN in self.df.columns:
                 output_cols.append(self.config.DATE_COLUMN)
            output_cols.extend(['sentiment', 'neg_score', 'neu_score', 'pos_score'])

            # Filter columns that actually exist in the dataframe before saving
            final_output_cols = [col for col in output_cols if col in self.df.columns]
            self.df[final_output_cols].to_csv(output_path, index=False, encoding='utf-8')
            print(f"Results saved successfully to {output_path}")
        except Exception as e:
            print(f"Error saving results to CSV '{output_path}': {e}")


    def run_pipeline(self):
        """Executes the full analysis pipeline step-by-step."""
        try:
            self.load_and_prep_data()
            self.process_text()
            self.run_sentiment_analysis()
            self.generate_visualizations()
            self.analyze_keyword_subsets()
            self.save_results()
            print("\n--- Pipeline execution finished successfully! ---")
        except Exception as e:
            print(f"\n--- Pipeline execution failed! ---")
            print(f"Error: {e}")
            # Consider adding more detailed error logging here if needed