# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "chardet",
#   "scikit-learn",
#   "rich",
#   "tenacity",
#   "openai",
#   "tabulate"
# ]
# ///

import os
import sys
import re
import json
import base64
import subprocess
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from dateutil import parser
import chardet
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from tabulate import tabulate
import logging

# Initialize console for rich logging
console = Console()

# Configure logging for tenacity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variable for AI Proxy token
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it before running the script.")

def retry_settings_with_logging():
    """Configures retry mechanism with logging."""
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )

@retry_settings_with_logging()
def detect_encoding(file_path):
    """Detect the encoding of a CSV file."""
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result['encoding']

@retry_settings_with_logging()
def read_csv(file_path):
    """Read a CSV file with automatic encoding detection and flexible date parsing."""
    try:
        console.log("Detecting file encoding...")
        encoding = detect_encoding(file_path)
        console.log(f"Detected encoding: {encoding}")

        df = pd.read_csv(file_path, encoding=encoding, encoding_errors='replace')

        # Enhanced date parsing
        for column in df.columns:
            if df[column].dtype == object and is_date_column(df[column]):
                console.log(f"Parsing dates in column: {column}")
                df[column] = df[column].apply(parse_date_with_regex)

        return df

    except Exception as e:
        console.log(f"[red]Error reading the file: {e}[/]")
        sys.exit(1)

def parse_date_with_regex(date_str):
    """Advanced date parsing with multiple strategies."""
    if not isinstance(date_str, str):
        return date_str

    if not re.search(r'\d', date_str):
        return np.nan

    patterns = [
        (r"\d{2}-[A-Za-z]{3}-\d{4}", "%d-%b-%Y"),
        (r"\d{2}-[A-Za-z]{3}-\d{2}", "%d-%b-%y"),
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
        (r"\d{2}/\d{2}/\d{4}", "%m/%d/%Y"),
        (r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y"),
    ]

    for pattern, date_format in patterns:
        if re.match(pattern, date_str):
            try:
                return pd.to_datetime(date_str, format=date_format, errors='coerce')
            except Exception as e:
                console.log(f"Date parsing error: {date_str}, {e}")
                return np.nan

    try:
        return parser.parse(date_str, fuzzy=True, dayfirst=False)
    except Exception as e:
        console.log(f"Fuzzy date parsing error: {date_str}, {e}")
        return np.nan

def is_date_column(column):
    """Enhanced date column detection."""
    if isinstance(column, str):
        if any(keyword in column.lower() for keyword in ['date', 'time', 'timestamp']):
            return True

    sample_values = column.dropna().head(10)
    date_patterns = [
        r"\d{2}-[A-Za-z]{3}-\d{2}", 
        r"\d{2}-[A-Za-z]{3}-\d{4}", 
        r"\d{4}-\d{2}-\d{2}", 
        r"\d{2}/\d{2}/\d{4}"
    ]

    for value in sample_values:
        if isinstance(value, str):
            for pattern in date_patterns:
                if re.match(pattern, value):
                    return True
    return False

def clean_data(data):
    """Enhanced data cleaning with more robust strategies."""
    console.log("[cyan]Advanced Data Cleaning...")
    data = data.drop_duplicates()
    
    # Identify numeric and categorical columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    # Fill missing values differently for numeric and categorical
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
    
    return data

def detect_outliers(data, contamination_rate=0.05):
    """Advanced outlier detection with configurable contamination."""
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        console.log("[yellow]No numeric data for outlier detection.")
        return data

    console.log("[cyan]Performing Advanced Outlier Detection...")
    
    # Use Isolation Forest with adaptable contamination
    model = IsolationForest(
        contamination=contamination_rate, 
        random_state=42,
        max_samples='auto',
        bootstrap=True
    )
    
    outliers = model.fit_predict(numeric_data)
    data['Outlier'] = (outliers == -1)
    
    # Log outlier statistics
    outlier_count = sum(data['Outlier'])
    console.log(f"[yellow]Detected {outlier_count} outliers out of {len(data)} records.")
    
    return data

def perform_clustering(data, n_clusters=3):
    """Adaptive clustering with dynamic cluster determination."""
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.shape[1] < 2:
        console.log("[yellow]Insufficient numeric features for clustering.")
        return data

    console.log("[cyan]Performing Adaptive Clustering...")
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Use adaptive number of clusters based on data size
    adapted_clusters = min(n_clusters, len(numeric_data) // 10)
    
    kmeans = KMeans(
        n_clusters=adapted_clusters, 
        random_state=42, 
        n_init='auto',
        algorithm='elkan'
    )
    
    data['Cluster'] = kmeans.fit_predict(scaled_data)
    
    return data

def query_llm(prompt, max_tokens=300, temperature=0.7):
    """Enhanced LLM query with token and creativity control."""
    try:
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a precise, insightful data analysis assistant. "
                               "Provide concise, structured insights. Use bullet points. "
                               "Focus on the most significant findings."
                },
                {"role": "user", "content": prompt},
            ],
        }
        
        payload_json = json.dumps(payload)
        result = subprocess.run(
            ["curl", "-X", "POST", url, 
             "-H", f"Authorization: Bearer {AIPROXY_TOKEN}", 
             "-H", "Content-Type: application/json", 
             "-d", payload_json],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            response_data = json.loads(result.stdout)
            return response_data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error in curl request: {result.stderr}")
    
    except Exception as e:
        console.log(f"[red]Error querying AI Proxy: {e}[/]")
        return "Error: Unable to generate narrative."

def create_multi_stage_narrative(analysis, visualizations):
    """Multi-stage narrative generation with targeted LLM interactions."""
    # Stage 1: Initial Overview
    overview_prompt = (
        f"Provide a high-level overview of this dataset:\n"
        f"Total Records: {analysis['shape'][0]}\n"
        f"Features: {', '.join(analysis['columns'].keys())}\n"
        "Focus on potential insights and interesting patterns."
    )
    overview = query_llm(overview_prompt, max_tokens=200)
    
    # Stage 2: Detailed Analysis
    detailed_prompt = (
        f"Based on the dataset overview and these summary statistics:\n"
        f"{json.dumps(analysis['summary_statistics'], indent=2)}\n\n"
        "Extract the TOP 3 most significant insights. "
        "Explain why these insights matter and potential actions."
    )
    detailed_insights = query_llm(detailed_prompt, max_tokens=300)
    
    # Stage 3: Predictive Narrative
    predictive_prompt = (
        "Considering the current dataset characteristics, "
        "suggest potential predictive models or further analyses "
        "that could provide valuable business or research insights."
    )
    predictive_suggestions = query_llm(predictive_prompt, max_tokens=200)
    
    # Combine stages
    full_narrative = f"""
    # Comprehensive Data Analysis Report

    ## Overview
    {overview}

    ## Key Insights
    {detailed_insights}

    ## Future Directions
    {predictive_suggestions}
    """
    
    return full_narrative

def main():
    console.log("[cyan]Starting Advanced Data Analysis Script...")
    
    if len(sys.argv) != 2:
        console.log("[red]Usage: python advanced_data_analysis.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    console.log(f"[yellow]Analyzing file: {file_path}[/]")
    
    df = read_csv(file_path)
    output_dir = create_output_folder(file_path)
    
    # Adaptive preprocessing
    df = clean_data(df)
    df = detect_outliers(df)
    df = perform_clustering(df)
    
    analysis = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "summary_statistics": df.describe(include="all").to_dict(),
    }
    
    # Visualization and analysis
    pca_path = perform_pca(df)
    visualizations = visualize_data(df, output_dir)
    visualizations.append(pca_path)
    
    # Multi-stage narrative generation
    story = create_multi_stage_narrative(analysis, visualizations)
    
    # Save comprehensive results
    save_results(output_dir, analysis, visualizations, story)
    
    console.log("[green]Advanced Analysis Completed Successfully!")

# Rest of the functions remain the same as in the previous script...

if __name__ == "__main__":
    main()
