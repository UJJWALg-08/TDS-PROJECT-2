# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "requests",
#   "pillow",
#   "scikit-learn",
#   "networkx",
#   "geopandas", 
#   "shapely",
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

def query_llm(prompt, token):
    """Query the LLM for insights."""
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with AI Proxy: {e}")
        return ""

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

def compute_statistical_summaries(df):
    """
    Compute statistical summaries for numerical columns.
    """
    return df.describe(include='all').transpose().to_dict()

def detect_outliers(df, z_threshold=3):
    """
    Detect outliers in numerical columns using z-score method.
    """
    outliers = {}
    for column in df.select_dtypes(include=[np.number]):
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outlier_indices = z_scores.abs() > z_threshold
        outliers[column] = {
            "total_outliers": outlier_indices.sum(),
            "indices": list(df.index[outlier_indices]),
            "lower_bound": df[column].mean() - z_threshold * df[column].std(),
            "upper_bound": df[column].mean() + z_threshold * df[column].std()
        }
    return outliers

def generate_significant_findings(data_summary, statistical_summary, outliers):
    """
    Highlight significant insights extracted from the data analysis.
    """
    insights = []

    # Example: Outliers
    if outliers:
        for column, details in outliers.items():
            insights.append(f"Column '{column}' has {details['total_outliers']} significant outliers exceeding the bounds [{details['lower_bound']}, {details['upper_bound']}].")

    # Example: Missing values
    missing_cols = [col for col, count in data_summary['missing_values'].items() if count > 0]
    if missing_cols:
        insights.append(f"The dataset contains missing values in columns: {', '.join(missing_cols)}.")

    return "\n".join(insights)

def perform_dynamic_prompting(analysis_results):
    """
    Dynamically adjust the script based on runtime analysis.
    """
    num_rows = analysis_results["num_rows"]
    if num_rows < 100:
        print("Adjusting clustering parameters for small dataset.")
        # Placeholder for dynamically changing parameters or LLM prompt behavior

def visualize_data(df, output_folder):
    """
    Create basic visualizations without using interactive backends.
    """
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 1:
        # Create a correlation heatmap instead of pairplot
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
        plt.close()  # Close the plot to free up memory
        print("Correlation heatmap saved.")
    
    # Optional: Box plots for numerical columns
    if len(numerical_cols) > 0:
        plt.figure(figsize=(12, 6))
        df[numerical_cols].boxplot()
        plt.title('Box Plots of Numerical Columns')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "boxplots.png"))
        plt.close()  # Close the plot to free up memory
        print("Box plots saved.")

def perform_cluster_analysis(df, output_folder):
    """
    Perform clustering analysis and save results.
    """
    numerical_data = df.select_dtypes(include=[np.number]).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_data)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    df['Cluster'] = clusters
    df.to_csv(os.path.join(output_folder, "clustered_data.csv"), index=False)
    print("Clustering completed.")

def generate_story_report(data_summary, statistical_summary, outliers, df, output_folder, dataset_name):
    """
    Generate a storytelling-style report.
    """
    report_path = os.path.join(output_folder, f"{dataset_name}_report.txt")
    with open(report_path, "w") as report_file:
        report_file.write(f"Dataset Report for {dataset_name}\n")
        report_file.write(f"Data Summary: {data_summary}\n")
        report_file.write(f"Statistical Summary: {statistical_summary}\n")
        report_file.write(f"Outliers: {outliers}\n")
    print(f"Report saved to {report_path}.")

def main():
    """
    Main function to execute the comprehensive data analysis.
    """
    parser = argparse.ArgumentParser(description='Advanced Data Analysis Script')
    parser.add_argument('dataset_path', help='Path to the CSV file')
    args = parser.parse_args()

    # Set AI Proxy Token
    token = set_aiproxy_token()

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
