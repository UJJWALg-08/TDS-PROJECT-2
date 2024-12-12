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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

# Validate command-line arguments
if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py <dataset.csv>")
    sys.exit(1)

dataset_file = sys.argv[1]
dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
output_dir = dataset_name

if not os.path.isfile(dataset_file):
    print(f"File {dataset_file} not found.")
    sys.exit(1)

try:
    ai_proxy_token = os.environ["AIPROXY_TOKEN"]
    print(f"AIPROXY_TOKEN detected: {ai_proxy_token[:10]}...")
except KeyError:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

try:
    data = pd.read_csv(dataset_file)
    print(f"Dataset {dataset_file} loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)# --- FILE SANITIZATION ---
def sanitize_filename(filename):
    return filename.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")


# --- DATE PARSING ---
def parse_dates(data, date_col):
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

# --- PERFORM GENERIC ANALYSIS ---
def perform_generic_analysis(data):
    analysis = {
        "shape": data.shape,
        "missing_values": data.isnull().sum().to_dict(),
        "dtypes": data.dtypes.astype(str).to_dict(),
        "head": data.head().to_dict(orient="list"),
    }
    numeric_cols = data.select_dtypes(include=["number"])
    if not numeric_cols.empty:
        analysis["correlations"] = numeric_cols.corr().to_dict()
    return analysis

# --- VISUALIZATION FUNCTIONS ---
def visualize_correlation_matrix(corr_matrix, filename):
    """Visualize the correlation matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300, metadata={'Title': 'No Text Rendering'})
    plt.close()

def visualize_numeric_distributions(data, column, filename):
    """Visualize numeric column distribution."""
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column].dropna(), kde=True, bins=30, color="blue")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300, metadata={'Title': 'No Text Rendering'})
    plt.close()

def visualize_categorical_counts(data, column, filename):
    """Visualize top categories in a column."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=data[column].value_counts().index[:10], y=data[column].value_counts().values[:10], palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top Categories in {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300, metadata={'Title': 'No Text Rendering'})
    plt.close()

# --- ADVANCED ANALYSES ---
def perform_clustering(data, numeric_cols, output_dir):
    normalized_data = data[numeric_cols].dropna()
    if normalized_data.empty:
        return None
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(normalized_data)
    data.loc[:, 'Cluster'] = clusters

    # Save clustering visualization
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=normalized_data.iloc[:, 0], y=normalized_data.iloc[:, 1], hue=clusters, palette="viridis")
    plt.title("Clustering Analysis")
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    filename = os.path.join(output_dir, "clustering.png")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300, metadata={'Title': 'No Text Rendering'})
    plt.close()
    return filename

def detect_outliers(data, numeric_cols, output_dir):
    """Detect and visualize outliers in numeric columns."""
    if not numeric_cols:  # Check if the list is empty
        print("No numeric columns available for outlier detection.")
        return None

    # Use Isolation Forest for outlier detection
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    try:
        data['Outlier'] = iso_forest.fit_predict(data[numeric_cols].fillna(0))
    except ValueError as e:
        print(f"Error in Isolation Forest: {e}")
        return

    for col in numeric_cols:
        if data[col].notnull().sum() < 2:  # Ensure the column has enough valid data
            print(f"Skipping boxplot for {col} due to insufficient data.")
            continue
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[col])
        plt.title(f"Outlier Detection - {col}")
        filename = os.path.join(output_dir, f"{sanitize_filename(col)}_outliers.png")
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300, metadata={'Title': 'No Text Rendering'})
        plt.close()
    
def time_series_analysis(data, date_col, numeric_col, output_dir):
    if date_col in data.columns and numeric_col in data.columns:
        try:
            data[date_col] = pd.to_datetime(data[date_col], format="%d-%b-%y", errors="coerce")
        except ValueError:
            print(f"Date format not consistent. Using auto-parsing for {date_col}.")
            data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

        data = data.dropna(subset=[date_col, numeric_col]).sort_values(by=date_col)

        if data.empty:
            print(f"No valid data points for time series analysis on {date_col} and {numeric_col}.")
            return None

        data["Rolling_Avg"] = data[numeric_col].rolling(window=30, min_periods=1).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(data[date_col], data["Rolling_Avg"], label="Rolling Average", color="blue", linewidth=2)
        plt.scatter(data[date_col], data[numeric_col], label="Original Data", color="lightblue", alpha=0.5, s=10)
        plt.title(f"Time Series Analysis of {numeric_col}")
        plt.xlabel("Date")
        plt.ylabel(numeric_col)
        plt.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        filename = os.path.join(output_dir, f"time_series_{sanitize_filename(numeric_col)}.png")
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()
        return filename
    else:
        print(f"Columns {date_col} and/or {numeric_col} not found in the dataset. Skipping time series analysis.")
        return None

def network_analysis(data, col1, col2, output_dir):
    if col1 in data.columns and col2 in data.columns:
        edges = data[[col1, col2]].dropna().values
        if len(edges) == 0:
            print(f"No valid edges found for {col1} and {col2}. Skipping network analysis.")
            return None

        graph = nx.Graph()
        graph.add_edges_from(edges)

        pos = nx.spring_layout(graph, k=0.5)

        plt.figure(figsize=(12, 12))
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=300,
            font_size=10,
            font_color="darkblue",
            node_color="skyblue",
            edge_color="gray",
            linewidths=0.5,
            alpha=0.8,
        )

        filename = os.path.join(output_dir, "network_analysis.png")
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()
        return filename
    else:
        print(f"Columns {col1} and/or {col2} not found in the dataset. Skipping network analysis.")
        return None

def create_visualizations(data, data_summary, output_dir):
    """Generate visualizations and save them in the output directory."""
    charts = []

    # Numeric and Categorical Columns
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    # Correlation Matrix
    if "correlations" in data_summary:
        corr_matrix = pd.DataFrame(data_summary["correlations"])
        if not corr_matrix.empty:
            filename = os.path.join(output_dir, "correlation_matrix.png")
            visualize_correlation_matrix(corr_matrix, filename)
            charts.append(filename)

    # Numeric Distributions
    for col in numeric_cols[:3]:
        filename = os.path.join(output_dir, f"{sanitize_filename(col)}_distribution.png")
        visualize_numeric_distributions(data, col, filename)
        charts.append(filename)

    # Categorical Counts
    for col in categorical_cols[:3]:
        filename = os.path.join(output_dir, f"{sanitize_filename(col)}_bar_chart.png")
        visualize_categorical_counts(data, col, filename)
        charts.append(filename)

    # Clustering Analysis
    clustering_chart = perform_clustering(data, numeric_cols[:2], output_dir)
    if clustering_chart:
        charts.append(clustering_chart)

    # Outlier Detection
    detect_outliers(data, numeric_cols, output_dir)

    # Time Series Analysis
    date_col = next((col for col in categorical_cols if "date" in col.lower()), None)
    if date_col:
        for col in numeric_cols[:1]:  # Perform time series on the first numeric column
            ts_chart = time_series_analysis(data, date_col, col, output_dir)
            if ts_chart:
                charts.append(ts_chart)

    # Network Analysis
    if len(categorical_cols) >= 2:
        network_chart = network_analysis(data, categorical_cols[0], categorical_cols[1], output_dir)
        if network_chart:
            charts.append(network_chart)
   
    return charts


# --- RESIZE IMAGES ---
def resize_images(image_files, size=(512, 512)):
    """Resize images and sanitize filenames."""
    resized_files = []
    for file in image_files:
        sanitized_file = file.replace(" ", "_").replace("(", "").replace(")", "")
        if sanitized_file != file and not os.path.exists(sanitized_file):
            os.rename(file, sanitized_file)

        try:
            img = Image.open(sanitized_file)
            img = img.resize(size)
            resized_file = sanitized_file.replace(".png", "_resized.png")
            img.save(resized_file)
            resized_files.append(resized_file)
        except Exception as e:
            print(f"Error resizing image {file}: {e}")
    return resized_files

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

# --- GENERATE STORY ---
def generate_story(data_summary, insights, resized_charts, output_dir, token):
    """Generate a Markdown story using data summary and insights."""
    chart_refs = "\n".join([
        f"![Chart]({os.path.relpath(chart, start=output_dir)})".replace("\\", "/") for chart in resized_charts
    ])
    
    # Add summaries for different types of analyses
    if any("bar_chart" in chart for chart in resized_charts):
        insights += """
        Bar graph analysis highlights the categorical distributions in the dataset, helping identify prominent categories and their frequencies.
        """
    
    if any("clustering" in chart for chart in resized_charts):
        insights += """
        Clustering analysis reveals natural groupings in the data, uncovering underlying patterns and similarities among data points.
        """
    
    if any("outliers" in chart for chart in resized_charts):
        insights += """
        Outlier detection identifies anomalies and deviations in numeric features, highlighting significant data points for deeper analysis.
        """
    
    if any("time_series" in chart for chart in resized_charts):
        insights += """
        Time series analysis uncovers trends and patterns over time, providing a temporal perspective on the dataset.
        """
    
    if any("network_analysis" in chart for chart in resized_charts):
        insights += """
        Network analysis visualizes relationships and connections between entities, offering insights into structural dependencies within the data.
        """
    
    if any("geographic_analysis" in chart for chart in resized_charts):
        insights += """
        Geographic analysis provides a spatial perspective by mapping data across regions or boundaries, uncovering geographic trends and disparities.
        """

    prompt = f"""
    Write a Markdown report:
    - Data Summary: {data_summary}
    - Insights: {insights}
    Include these charts: {chart_refs}
    """
    return query_llm(prompt, token)

# --- SAVE README ---
def save_readme(content, output_dir):
    """Save README.md."""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(content)
    print(f"README.md created in {output_dir}.")

# Main workflow
# Preprocess data and ensure date parsing is handled
data_summary = perform_generic_analysis(data)  # Perform analysis on preprocessed data
charts = create_visualizations(data, data_summary, output_dir)  # Visualize the preprocessed data

# Ensure images are resized and sanitized
resized_charts = resize_images(charts)

# Query the LLM for insights
insights = query_llm(f"Analyze this dataset summary: {data_summary}", ai_proxy_token)

# Generate the story
story = generate_story(data_summary, insights, resized_charts, output_dir, ai_proxy_token)

# Save README.md
save_readme(story, output_dir)

print(f"Analysis complete. Outputs saved in {output_dir}.")
