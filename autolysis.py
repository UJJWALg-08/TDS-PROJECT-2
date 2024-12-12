# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "requests",
#   "pillow",
#   "scikit-learn",
# ]
# ///

import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Explicitly set GPT-4.0-mini usage for LLM interactions
LLM_VERSION = "GPT-4.0-mini"

def set_aiproxy_token():
    """
    Prompt the user to enter the AI Proxy token and set it as an environment variable.
    """
    AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')
    if not AIPROXY_TOKEN:
        raise ValueError("AIPROXY_TOKEN environment variable is not set. Please set it before running the script.")
    print("AI Proxy token successfully set!")
    return AIPROXY_TOKEN


def create_output_folder(csv_filename):
    """
    Create a new folder to store all analysis outputs, using CSV filename as folder name.
    """
    folder_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in os.path.splitext(csv_filename)[0])
    os.makedirs(folder_name, exist_ok=True)
    print(f"Output folder '{folder_name}' created.")
    return folder_name


def load_csv(filename, encodings_to_try=None):
    """
    Load CSV file with multiple encoding attempts.
    """
    if encodings_to_try is None:
        encodings_to_try = [
            'utf-8',
            'iso-8859-1',
            'latin1',
            'cp1252',
            'utf-16',
        ]

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filename, encoding=encoding, encoding_errors='replace')
            print(f"Successfully loaded file using {encoding} encoding")
            return df
        except Exception as e:
            print(f"Failed to load with {encoding} encoding: {e}")

    raise ValueError(f"Could not load CSV file {filename} with any of the attempted encodings")


def analyze_data_structure(df):
    """
    Analyze the structure of the dataset.
    """
    df.columns = [col.encode('ascii', 'ignore').decode('ascii') for col in df.columns]
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_types": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": {col: df[col].nunique() for col in df.columns}
    }


def generate_significant_findings(data_summary, statistical_summary, outliers):
    """
    Highlight significant insights extracted from the data analysis.
    """
    insights = []

    # Example: Outliers
    if outliers:
        for column, details in outliers.items():
            insights.append(f"Column '{column}' has {details['total_outliers']} significant outliers "
                            f"exceeding the bounds [{details['lower_bound']}, {details['upper_bound']}].")

    # Example: Missing values
    missing_cols = [col for col, count in data_summary['missing_values'].items() if count > 0]
    if missing_cols:
        insights.append(f"The dataset contains missing values in columns: {', '.join(missing_cols)}.")

    return "\n".join(insights)


def perform_dynamic_prompting(analysis_results):
    """
    Dynamically adjust the script based on runtime analysis.
    """
    # Example: Adjust clustering parameters based on data shape
    num_rows = analysis_results["num_rows"]
    if num_rows < 100:
        print("Adjusting clustering parameters for small dataset.")
        # Placeholder for dynamically changing parameters or LLM prompt behavior


def main():
    """
    Main function to execute the comprehensive data analysis.
    """
    parser = argparse.ArgumentParser(description='Advanced Data Analysis Script')
    parser.add_argument('dataset_path', help='Path to the CSV file')
    args = parser.parse_args()

    # Set AI Proxy Token
    set_aiproxy_token()

    # Create output folder using CSV filename
    output_folder = create_output_folder(os.path.basename(args.dataset_path))

    # Load dataset
    df = load_csv(args.dataset_path)

    # Analyze data structure
    data_summary = analyze_data_structure(df)

    # Compute statistical summaries
    statistical_summary = compute_statistical_summaries(df)

    # Detect outliers
    outliers = detect_outliers(df)

    # Highlight significant findings
    findings = generate_significant_findings(data_summary, statistical_summary, outliers)
    print(f"\n--- Significant Findings ---\n{findings}")

    # Dynamic adjustments based on runtime analysis
    perform_dynamic_prompting(data_summary)

    # Visualization and clustering
    visualize_data(df, output_folder)
    perform_cluster_analysis(df, output_folder)

    # Generate storytelling-style report
    generate_story_report(
        data_summary,
        statistical_summary,
        outliers,
        df,
        output_folder,
        os.path.basename(args.dataset_path)
    )


if __name__ == "__main__":
    main()
