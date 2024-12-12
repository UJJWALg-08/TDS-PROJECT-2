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
    # Remove .csv extension and replace any problematic characters
    folder_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in os.path.splitext(csv_filename)[0])
    os.makedirs(folder_name, exist_ok=True)
    print(f"Output folder '{folder_name}' created.")
    return folder_name


def load_csv(filename, encodings_to_try=None):
    """
    Load CSV file with multiple encoding attempts
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
    Analyze the structure of the dataset
    """
    # Normalize column names to remove non-ASCII characters
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
    Compute statistical summaries and correlation matrix
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_columns].describe().to_dict()
    correlation_matrix = df[numeric_columns].corr()
    return {
        "summary_statistics": summary,
        "correlation_matrix": correlation_matrix
    }


def detect_outliers(df):
    """
    Detect outliers in numeric columns using IQR method
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        if len(column_outliers) > 0:
            outliers[column] = {
                "total_outliers": len(column_outliers),
                "percentage": (len(column_outliers) / len(df)) * 100,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
    return outliers


def perform_cluster_analysis(df, output_folder):
    """
    Perform cluster analysis using KMeans and PCA
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_columns]

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Visualization
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.title('Cluster Analysis (PCA Visualization)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cluster_analysis.png'), dpi=150)
    plt.close()

    return clusters


def visualize_data(df, output_folder):
    """
    Generate comprehensive visualizations, limited to 3 png files
    """
    print("\n--- Generating Visualizations ---")

    # Numeric column distributions
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=numeric_df.iloc[:, :3], kde=True)
        plt.title("Distribution of Numeric Columns")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'numeric_distributions.png'), dpi=150)
        plt.close()

    # Correlation heatmap
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'correlation_heatmap.png'), dpi=150)
        plt.close()


def generate_story_report(data_summary, statistical_summary, outliers, df, output_folder, input_filename):
    """
    Generate a storytelling-style analysis report in Markdown format
    """

    # Safely remove non-printable characters and handle potential encoding issues
    def clean_text(text):
        if isinstance(text, str):
            return ''.join(char for char in text if char.isprintable())
        return str(text)

    # Prepare the story with cleaned and safe text
    story = f"""# Data Expedition: Unveiling Hidden Insights 🕵️‍♀️🔍

## The Journey Begins 🚀
On a crisp morning, we embarked on an analytical adventure with **{clean_text(input_filename)}**, a dataset brimming with untold stories. Our mission: to transform raw numbers into meaningful narratives.

## The Landscape of Data 🗺️
Imagine a terrain of information:
- **{len(df)} explorers (rows)** traversing **{len(df.columns)} paths (columns)**
- Numeric territories mapped: {', '.join(clean_text(col) for col in df.select_dtypes(include=[np.number]).columns)}

## Whispers of the Data 🤫
Our expedition uncovered fascinating signals:

### The Structure Speaks
- **Diversity of Information**: {len(df.columns)} unique data streams
- **Missing Whispers**: Columns with untold parts of the story:
  {' | '.join([f"{clean_text(col)} (silent for {count} moments)" for col, count in data_summary['missing_values'].items() if count > 0])}

### Statistical Echoes 📊
As we delved deeper, the data revealed its rhythms and patterns. Some columns danced wildly, others hummed quietly.

## Unexpected Discoveries 🔮
- **Outliers**: Rare data points that challenge the norm
- **Correlation Secrets**: Hidden connections between variables (see our correlation heatmap)

## The Visual Chronicles 📸
We've captured three moments of our journey:
1. **numeric_distributions.png**: The rhythm of our numeric explorers
2. **correlation_heatmap.png**: The intricate web of data relationships
3. **cluster_analysis.png**: Revealing natural groupings within our data

## Final Reflections 🌅
Every dataset tells a story. Ours spoke of complexity, patterns, and the beautiful unpredictability of data.

*Expedition log closed on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    # Write story to README.md with UTF-8 encoding
    report_path = os.path.join(output_folder, "README.md")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(story)

    print(f"Data story generated at {report_path}")


def main():
    """
    Main function to execute the comprehensive data analysis
    """
    # Setup argument parsing
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

    # Visualize data
    visualize_data(df, output_folder)

    # Perform cluster analysis
    perform_cluster_analysis(df, output_folder)

    # Generate story report
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