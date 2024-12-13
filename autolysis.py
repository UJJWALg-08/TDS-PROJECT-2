# /// script
# requires-python = ">=3.8, <3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "chardet",
#     "requests",
#     "seaborn",
#     "matplotlib",
#     "python-dotenv",
#     "missingno"
# ]
# ///

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import requests
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore
import missingno as msno
import argparse
import chardet

# Set the AIPROXY_TOKEN environment variable if not already set
if "AIPROXY_TOKEN" not in os.environ:
    api_key = input("Please enter your OpenAI API key: ")
    os.environ["AIPROXY_TOKEN"] = api_key

api_key = os.environ["AIPROXY_TOKEN"]
API_BASE = "https://aiproxy.sanand.workers.dev/openai/v1"

# Function to detect file encoding
def detect_encoding(filename):
    with open(filename, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    
    # Handle UTF-16 without BOM
    if result['encoding'] == 'UTF-16' and not raw_data.startswith(b'\xff\xfe') and not raw_data.startswith(b'\xfe\xff'):
        result['encoding'] = 'utf-16'  # Assume UTF-16 if BOM is missing
    
    return result['encoding']

def load_and_clean_data(filename):
    try:
        encoding = detect_encoding(filename)
        df = pd.read_csv(filename, encoding=encoding)

        # Drop rows with all NaN values
        df.dropna(axis=0, how='all', inplace=True)

        # Fill missing values in numeric columns with the mean of the column
        numeric_columns = df.select_dtypes(include='number')
        df[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.mean())

        # Handle missing values in non-numeric columns (e.g., fill with 'Unknown')
        non_numeric_columns = df.select_dtypes(exclude='number')
        df[non_numeric_columns.columns] = non_numeric_columns.fillna('Unknown')

        return df

    except UnicodeDecodeError as e:
        print(f"Error: Could not decode the file. Ensure it is properly encoded. Details: {e}")
        sys.exit(1)


# Function to summarize the dataset
def summarize_data(df):
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'types': df.dtypes.to_dict(),
        'descriptive_statistics': df.describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return summary

# Outlier detection function using Z-Score
def detect_outliers(df):
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_df))
    outliers = (z_scores > 3).sum(axis=0)
    outlier_info = {
        column: int(count) for column, count in zip(numeric_df.columns, outliers)
    }
    return outlier_info

# Correlation analysis function
def correlation_analysis(df):
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()
    return correlation_matrix.to_dict()

# Cluster analysis using KMeans
def perform_clustering(df, n_clusters=3):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    return df, kmeans

# PCA for dimensionality reduction (optional)
def perform_pca(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)
    df['PCA1'] = pca_components[:, 0]
    df['PCA2'] = pca_components[:, 1]
    return df

# Function to create visualizations
def create_visualizations(df):
    visualizations = []  # Initialize an empty list to store image paths

    # Visualization for missing data
    try:
        msno.matrix(df)
        missing_img = 'missing_data.png'
        plt.savefig(missing_img, dpi=100)  # Low resolution
        plt.close()
        visualizations.append(missing_img)
    except Exception as e:
        print(f"Failed to generate missing data visualization: {e}")

    # Correlation heatmap
    try:
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] > 1:
            plt.figure(figsize=(10, 6), constrained_layout=True)
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Correlation Matrix")
            correlation_img = 'correlation_matrix.png'
            plt.savefig(correlation_img, dpi=100)  # Low resolution
            plt.close()
            visualizations.append(correlation_img)
    except Exception as e:
        print(f"Failed to generate correlation heatmap: {e}")

    # Cluster visualization (after performing PCA)
    try:
        if 'PCA1' in df.columns and 'Cluster' in df.columns:
            plt.figure(figsize=(8, 6), constrained_layout=True)
            sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set1')
            plt.title("Cluster Analysis (PCA)")
            cluster_img = 'cluster_analysis.png'
            plt.savefig(cluster_img, dpi=100)  # Low resolution
            plt.close()
            visualizations.append(cluster_img)
    except Exception as e:
        print(f"Failed to generate cluster analysis visualization: {e}")

    return visualizations  # Always return the list, even if empty


# Function to generate GPT-4o-Mini analysis story
def generate_analysis_story(summary, outliers, correlation_matrix):
    prompt = f"""
    Given the following dataset summary:
    - Shape: {summary['shape']}
    - Columns: {', '.join(summary['columns'])}
    - Data Types: {summary['types']}
    - Descriptive Statistics: {summary['descriptive_statistics']}
    - Missing values: {summary['missing_values']}

    Additionally, the outliers detected are: {outliers}

    The correlation matrix of the dataset is:
    {correlation_matrix}

    Generate a detailed, insightful analysis of the dataset. Include an interpretation of the statistics, correlations, outliers, and any recommendations for further analysis. Additionally, narrate the findings in a story-like manner to convey the insights effectively.
    """

    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(
        f"{API_BASE}/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,  # Low token count to reduce cost
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Function to write the README

def write_readme(story, visualizations, filename):
    with open('README.md', 'w') as f:
        f.write(f"# Dataset Analysis of {filename}\n")
        f.write("\n## Dataset Analysis Story\n")
        f.write(f"{story}\n")
        f.write("\n## Visualizations\n")
        for img in visualizations:
            f.write(f"![{img}]({img})\n")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Automated Dataset Analysis")
    parser.add_argument("csv_filename", help="Path to the CSV file to analyze")
    args = parser.parse_args()

    df = load_and_clean_data(args.csv_filename)
    summary = summarize_data(df)
    outliers = detect_outliers(df)
    correlation_matrix = correlation_analysis(df)
    df, _ = perform_clustering(df)
    df = perform_pca(df)
    visualizations = create_visualizations(df)  # Capture visualizations here
    story = generate_analysis_story(summary, outliers, correlation_matrix)
    write_readme(story, visualizations, args.csv_filename)  # Pass visualizations
    print("Analysis complete. Results saved in README.md.")

if __name__ == "__main__":
    main()
