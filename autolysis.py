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

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set OpenAI API Key from environment variable
openai.api_key = os.environ.get("AIPROXY_TOKEN")
if not openai.api_key:
    print("Error: Please set the AIPROXY_TOKEN environment variable.")
    sys.exit(1)


def query_llm(prompt, functions=None, model="gpt-4o-mini"):
    try:
        messages = [
            {"role": "system", "content": "You are an expert data scientist. Please follow my instructions."},
            {"role": "user", "content": prompt}
        ]
        kwargs = {}
        if functions:
          kwargs["functions"] = functions
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
             **kwargs,
        )
        if not response['choices'][0]['message'].get("function_call"):
          return response['choices'][0]['message']['content']
        else:
          function_call = response['choices'][0]['message']['function_call']
          return function_call
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded data from: {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Could not parse the CSV file {file_path}. Please ensure it is a valid CSV file")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

def generate_summary_stats(data):
    """Generates summary statistics and missing values for a DataFrame."""
    summary = data.describe(include='all').transpose()
    missing_values = data.isnull().sum()
    return summary, missing_values

def analyze_correlations(data):
    """Calculates the correlation matrix for numeric columns."""
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
      return None
    correlation_matrix = numeric_data.corr()
    return correlation_matrix

def detect_outliers(data):
    """Detects outliers using z-score."""
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
      return None
    outliers = numeric_data[(np.abs((numeric_data - numeric_data.mean()) / numeric_data.std()) > 3).any(axis=1)]
    return outliers

def detect_time_series_opportunity(data):
    """Determines if time-series analysis is applicable by checking for a date column."""
    date_columns = [col for col in data.columns if 'date' in col.lower()]
    if date_columns:
      return True
    return False

def perform_clustering(data):
    """Perform clustering analysis using K-means."""
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        return None, None

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    for n_clusters in range(2, min(11, scaled_data.shape[0])):
      kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
      cluster_labels = kmeans.fit_predict(scaled_data)
      score = silhouette_score(scaled_data, cluster_labels)
      silhouette_scores.append((n_clusters, score))
    
    if silhouette_scores:
        best_n_clusters = max(silhouette_scores, key=lambda item: item[1])[0]
        kmeans = KMeans(n_clusters=best_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        return kmeans, cluster_labels
    
    return None, None

def create_context_for_llm(data, summary, missing_values):
    """Generates a structured context for the LLM."""
    context = {
        "columns": [{"name": col, "type": str(data[col].dtype), "missing_values": missing_values[col]} for col in data.columns],
        "example_rows": data.head(3).to_dict(orient="records"),
        "shape": data.shape,
        "summary_statistics": summary.to_dict(),
    }
    return json.dumps(context, indent=2)


def generate_and_save_plots(data, file_prefix, correlation_matrix=None, outliers=None, cluster_labels=None, kmeans_model=None):
    """Generates and saves visualizations."""
    sns.set(style="whitegrid")

    # Correlation heatmap
    if correlation_matrix is not None and not correlation_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix")
        plt.savefig(f"{file_prefix}_correlation_matrix.png")
        plt.close()

    # Pairplot
    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        sns.pairplot(numeric_data)
        plt.savefig(f"{file_prefix}_pairplot.png")
        plt.close()

    # Outliers boxplot
    if outliers is not None and not outliers.empty:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data.select_dtypes(include=[np.number]), orient="h")
        plt.title("Outlier Detection")
        plt.savefig(f"{file_prefix}_outliers.png")
        plt.close()
    
    # Cluster scatterplot
    if cluster_labels is not None and kmeans_model is not None:
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] >= 2:
            scaled_data = StandardScaler().fit_transform(numeric_data)
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=cluster_labels, palette="viridis", legend="full")
            plt.title(f"Cluster Analysis (K={kmeans_model.n_clusters})")
            plt.savefig(f"{file_prefix}_clusters.png")
            plt.close()


def analyze_with_llm(context_str, file_prefix, has_time_series=False, correlation_matrix=None, outliers=None, cluster_labels=None, kmeans_model=None):
      """Analyzes the data using LLM, leveraging function calls for dynamic analysis."""
      
      functions = [
        {
            "name": "perform_time_series_analysis",
            "description": "Perform time series analysis if a date column is present and a time-based analysis is feasible.",
             "parameters": {
                  "type": "object",
                    "properties": {
                        "date_column": {
                            "type": "string",
                            "description": "Name of the date column in the dataset."
                         }
                        },
                       "required": ["date_column"],
              },
        },
        {
          "name": "generate_cluster_summary",
          "description": "Summarize the characteristics of identified clusters based on cluster labels",
          "parameters": {
             "type": "object",
             "properties": {
                "cluster_count": {
                     "type": "integer",
                     "description": "The number of clusters found during the analysis"
                     },
              },
             "required": ["cluster_count"],
              }
        },
        {
          "name": "generate_outlier_summary",
            "description": "Summarize the detected outliers. Highlight any unusual or interesting characteristics.",
             "parameters": {
                  "type": "object",
                    "properties": {
                         "outlier_count": {
                           "type": "integer",
                           "description": "The number of outliers detected"
                         }
                       },
                       "required": ["outlier_count"],
              }
         }
      ]
      prompt = f"""
        You are an expert data scientist.
        Analyze this dataset context:
        {context_str}

        Here is what I know about the dataset:

        - Time series analysis is {'applicable' if has_time_series else 'not applicable'}
        - Correlation matrix is {'available' if correlation_matrix is not None else 'not available'}
        - Outliers are {'detected' if outliers is not None else 'not detected'}
        - Clustering is {'performed' if cluster_labels is not None else 'not performed'}

        Based on this information provide:
         - a summary of the findings
         - recommendations for further analysis
         - a description of any significant patterns detected

         You may use any of the following functions to help you perform a more detailed analysis if required.
      """

      response = query_llm(prompt, functions=functions)
      
      analysis_summary = ""
      if not isinstance(response, dict):
         analysis_summary = response
      else:
          function_name = response["name"]
          function_args = json.loads(response["arguments"])
          if function_name == "perform_time_series_analysis":
            analysis_summary += "Performing a time series analysis..."
            analysis_summary += f"You asked me to analize the '{function_args['date_column']}' column but the requested functionality was not implemented in this version."
          elif function_name == "generate_cluster_summary":
             analysis_summary += f"Here is a summary of the generated clusters, based on the analysis we identified {function_args['cluster_count']} clusters."
             prompt_cluster = f"""
             Provide a summary of the clusters identified in the data:
             {context_str}
             The number of clusters is {function_args['cluster_count']}.
             """
             analysis_summary += query_llm(prompt_cluster)
          elif function_name == "generate_outlier_summary":
            analysis_summary += f"Here is a summary of the detected outliers, I found {function_args['outlier_count']} outliers."
            prompt_outliers = f"""
             Provide a summary of the outliers identified in the data:
             {context_str}
             The number of outliers is {function_args['outlier_count']}.
             """
            analysis_summary += query_llm(prompt_outliers)
          else:
             analysis_summary += "I couldn't understand your request, I'll proceed generating a summary without additional analysis."

      return analysis_summary


def generate_readme(file_path, summary, insights, file_prefix):
    """Generates and saves the README file."""
    
    readme_content = f"""# Analysis of {file_path}

## Dataset Summary

{summary}

## Key Insights

{insights}

## Visualizations

"""

    image_files = [
        f"{file_prefix}_correlation_matrix.png",
        f"{file_prefix}_pairplot.png",
        f"{file_prefix}_outliers.png",
        f"{file_prefix}_clusters.png"
    ]

    for image_file in image_files:
        if os.path.exists(image_file):
            readme_content += f"![{image_file.replace('.png','').replace(f'{file_prefix}_', '').replace('_', ' ').title()}]({image_file})\n"

    with open("README.md", "w") as f:
       f.write(readme_content)

def analyze_images_with_llm(file_prefix):
  """ Analyzes the generated plots using the LLM's vision capabilities"""
  vision_prompt = f"""
  I am providing you with the following plots, for the ones you are able to identify, please provide an analysis.
    
    - Correlation matrix: {file_prefix}_correlation_matrix.png
    - Pairplot: {file_prefix}_pairplot.png
    - Outliers: {file_prefix}_outliers.png
    - Clusters: {file_prefix}_clusters.png

  """
  messages = [{"role": "user", "content": vision_prompt}]
  if os.path.exists(f"{file_prefix}_correlation_matrix.png"):
      messages[0]["content"] = [{"type": "text", "text": vision_prompt}, {
              "type": "image_url",
              "image_url": {
                    "url": f"data:image/png;base64,{base64_encode_image(f'{file_prefix}_correlation_matrix.png')}"
                }
      }]
  if os.path.exists(f"{file_prefix}_pairplot.png"):
       messages[0]["content"].append({
              "type": "image_url",
              "image_url": {
                    "url": f"data:image/png;base64,{base64_encode_image(f'{file_prefix}_pairplot.png')}"
                }
      })
  if os.path.exists(f"{file_prefix}_outliers.png"):
        messages[0]["content"].append({
              "type": "image_url",
              "image_url": {
                    "url": f"data:image/png;base64,{base64_encode_image(f'{file_prefix}_outliers.png')}"
                }
      })
  if os.path.exists(f"{file_prefix}_clusters.png"):
        messages[0]["content"].append({
              "type": "image_url",
              "image_url": {
                    "url": f"data:image/png;base64,{base64_encode_image(f'{file_prefix}_clusters.png')}"
                }
      })
  try:
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        max_tokens=1024
    )
    return response['choices'][0]['message']['content']
  except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def base64_encode_image(image_path):
  """Encodes an image to base64"""
  import base64
  with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
  return encoded_string
    
def analyze_csv(file_path):
    """Main analysis function."""
    print(f"Starting analysis for: {file_path}")
    file_prefix = os.path.splitext(os.path.basename(file_path))[0]

    # Load data
    data = load_data(file_path)

    # Generate initial summaries
    summary, missing_values = generate_summary_stats(data)

    # Correlation analysis
    correlation_matrix = analyze_correlations(data)

    # Outlier detection
    outliers = detect_outliers(data)
    
    # Clustering
    kmeans_model, cluster_labels = perform_clustering(data)

    # Check for time-series
    has_time_series = detect_time_series_opportunity(data)

    # LLM Context
    context_str = create_context_for_llm(data, summary, missing_values)

    # LLM Analysis
    insights = analyze_with_llm(context_str, file_prefix, has_time_series, correlation_matrix, outliers, cluster_labels, kmeans_model)

    # Generate and save plots
    generate_and_save_plots(data, file_prefix, correlation_matrix, outliers, cluster_labels, kmeans_model)

    # Generate README
    generate_readme(file_path, summary, insights, file_prefix)

    # Analyze images using LLM Vision
    vision_insights = analyze_images_with_llm(file_prefix)

    if vision_insights:
        with open("README.md", "a") as f:
            f.write("\n## Image Analysis Insights\n")
            f.write(vision_insights)

    print("Analysis completed. Results are in README.md and generated PNG files.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <path_to_csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_csv(csv_file)
