"""
Main execution script for the Sentiment Analysis Pipeline.

This script imports the configuration, creates the pipeline instance,
and runs the full analysis workflow.
"""

# Import the pipeline class and configuration
from pipeline import SentimentAnalysisPipeline
import config # Import the config module directly

if __name__ == "__main__":
    print("=============================================")
    print("=== Sentiment Analysis Pipeline - Started ===")
    print("=============================================")

    # Create an instance of the pipeline, passing the config module
    analysis_pipeline = SentimentAnalysisPipeline(config)

    # Run the entire pipeline
    analysis_pipeline.run_pipeline()

    print("\n=============================================")
    print("=== Sentiment Analysis Pipeline - Finished ===")
    print("=============================================")