# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "scipy",
#   "scikit-learn",
#   "requests",
#   "chardet"
# ]
# ///
import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import chardet
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List


def json_serialize_handler(obj):

        #JSON serialization handler for various non-standard types

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()  # Convert NumPy scalar to Python scalar
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):
            return str(obj.dtype)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class AdvancedDataAnalyzer:
    def __init__(self, filename):

        # Initialize the data analyzer with robust error handling and dynamic configuration

        self.filename = filename
        self.df = self._load_csv()
        self.config = {
            'visualization_dpi': 512/8,
            'max_scatter_plots': 12,
            'clustering_methods': ['kmeans', 'dbscan'],
            'outlier_detection_methods': ['iqr', 'isolation_forest']
        }


    def _load_csv(self):

        # Enhanced CSV loading with comprehensive encoding detection and validation

        encodings_to_try = [
            'utf-8', 'iso-8859-1', 'latin1', 'cp1252', 'utf-16',
            'ascii', 'big5', 'shift_jis', 'gb2312'
        ]

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(self.filename, encoding=encoding,
                                 low_memory=False,
                                 parse_dates=True,
                                 infer_datetime_format=True)

                # Additional data validation
                if df.empty:
                    print(f"Warning: Empty dataframe loaded with {encoding} encoding")
                    continue

                print(f"Successfully loaded file using {encoding} encoding")
                return df
            except Exception as e:
                print(f"Failed to load with {encoding} encoding: {e}")

        raise ValueError("Could not load CSV file with any attempted encoding")

    def advanced_data_profiling(self):

        # Comprehensive data profiling with advanced statistical insights

        # Data type inference and advanced type detection
        data_types = {}
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)

            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                data_types[col] = 'datetime'
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                if unique_ratio < 0.05:
                    data_types[col] = 'categorical_numeric'
                else:
                    data_types[col] = 'continuous'
            elif pd.api.types.is_categorical_dtype(self.df[col]):
                data_types[col] = 'categorical'
            else:
                data_types[col] = 'text'

        # Advanced statistical tests
        normality_tests = {}
        for col, dtype in data_types.items():
            if dtype == 'continuous':
                try:
                    _, p_value = stats.normaltest(self.df[col].dropna())
                    normality_tests[col] = {
                        'is_normal': p_value > 0.05,
                        'p_value': p_value
                    }
                except Exception:
                    pass

        # Mutual information for feature importance
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_importance = {}
        for col in numeric_cols:
            try:
                importance = mutual_info_classif(
                    self.df[[col]],
                    pd.qcut(self.df[col], q=4, labels=False)
                )[0]
                feature_importance[col] = importance
            except Exception:
                pass

        return {
            'data_types': data_types,
            'normality_tests': normality_tests,
            'feature_importance': feature_importance
        }

    def advanced_outlier_detection(self):

        # Multi-method outlier detection with visualization

        # Dynamic Analyis
        outlier_methods = {
            'iqr': self._iqr_outliers,
            'isolation_forest': self._isolation_forest_outliers
        }

        all_outliers = {}
        for method_name, method_func in outlier_methods.items():
            outliers = method_func()
            if outliers:
                all_outliers[method_name] = outliers

        # Enhanced visualization
        plt.figure(figsize=(20, 10))
        plt.suptitle('Outlier Detection Comparison', fontsize=16)

        # Plotting the outliers
        for i, (method, method_outliers) in enumerate(all_outliers.items(), 1):
            plt.subplot(2, len(all_outliers), i)
            for col, details in method_outliers.items():
                sns.boxplot(data=self.df, y=col)
                plt.title(f'{method.replace("_", " ").title()}: {col} Outliers', fontsize=10)
                plt.xticks(rotation=45)

        # Optimize the image configuration
        plt.tight_layout()
        plt.savefig('outliers_boxplot.png', dpi=self.config['visualization_dpi'])
        plt.close()

        return all_outliers

    def _iqr_outliers(self):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}

        # Quantile calculations

        for column in numeric_columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Creating column outliers
            column_outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
            if len(column_outliers) > 0:
                outliers[column] = {
                    "total_outliers": len(column_outliers),
                    "percentage": (len(column_outliers) / len(self.df)) * 100,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }

        return outliers

    def _isolation_forest_outliers(self):
        # Dynamic Analyis
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) == 0:
            return {}

        X = self.df[numeric_columns]
        imputer = SimpleImputer(strategy='median')
        scaler = RobustScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        clf = IsolationForest(contamination=0.1, random_state=42)
        y_pred = clf.fit_predict(X_scaled)

        outliers = {}
        for col in numeric_columns:
            column_outliers = self.df[y_pred == -1]
            if len(column_outliers) > 0:
                outliers[col] = {
                    "total_outliers": len(column_outliers),
                    "percentage": (len(column_outliers) / len(self.df)) * 100
                }

        return outliers

    def advanced_clustering(self):

        # Multi-method clustering with optimal cluster determination

        numeric_columns = self.df.select_dtypes(include=[np.number]).columns

        # Data preparation
        X = self.df[numeric_columns]
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        # Dimensionality reduction
        pca = PCA(n_components=min(2, len(numeric_columns)))
        X_pca = pca.fit_transform(X_scaled)

        # Clustering methods
        clustering_results = {}

        # K-means with silhouette score for optimal clusters
        max_clusters = min(10, len(X) // 2)
        silhouette_scores = []

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append((n_clusters, score))

        optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]

        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        clustering_results['kmeans'] = {
            'labels': kmeans.fit_predict(X_scaled),
            'optimal_clusters': optimal_clusters
        }

        # DBSCAN clustering
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clustering_results['dbscan'] = {
                'labels': dbscan.fit_predict(X_scaled)
            }
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")

        # Visualization
        plt.figure(figsize=(15, 6))
        for i, (method, result) in enumerate(clustering_results.items(), 1):
            plt.subplot(1, len(clustering_results), i)
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                                  c=result['labels'],
                                  cmap='viridis',
                                  alpha=0.7)
            plt.title(f'{method.capitalize()} Clustering')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.colorbar(scatter, label='Cluster')

        # Optimize the image configuration
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=self.config['visualization_dpi'])
        plt.close()

        return clustering_results


    def generate_llm_summary(self) -> str:

        # Advanced LLM-powered narrative generation with multi-stage, contextual analysis with reduced token usage

        # Prepare comprehensive analysis context
        analysis_context = self._prepare_analysis_context()

        # Define a series of prompts for multi-faceted narrative generation

        narrative_stages = [
            self._generate_overview_prompt(analysis_context),
            self._generate_insights_prompt(analysis_context),
            self._generate_recommendations_prompt(analysis_context),
            self.generate_graph_suggestion(analysis_context)
        ]

        # Concatenate narratives from different stages
        full_narrative = self._compose_narrative(narrative_stages)

        return full_narrative

    def _prepare_analysis_context(self) -> Dict[str, Any]:

        # Prepare a comprehensive context for narrative generation
        # Reducing token usage
        # Combine insights from various analyses and utilizing multiple llms prompts logically

        profiling = self.advanced_data_profiling()
        outliers = self.advanced_outlier_detection()
        clustering = self.advanced_clustering()

        return {
            'dataset_overview': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'columns': list(self.df.columns)
            },
            'data_types': profiling['data_types'],
            'normality_tests': profiling['normality_tests'],
            'feature_importance': profiling['feature_importance'],
            'outliers': outliers,
            'clustering': {
                'method': 'K-means',
                'optimal_clusters': clustering['kmeans']['optimal_clusters'],
                'labels': clustering['kmeans']['labels'].tolist()
            },
            'summary_statistics': self.df.describe().to_dict()
        }

    def _generate_overview_prompt(self, context: Dict[str, Any]) -> str:

        # Generate an overview prompt focusing on dataset structure
        # Dynamic prompting according to the data
        # Providing optimum amount of data
        overview_prompt = f"""
        Provide a comprehensive overview of the dataset:

        Dataset Structure:
        - Total Rows: {context['dataset_overview']['total_rows']}
        - Total Columns: {context['dataset_overview']['total_columns']}
        - Columns: {', '.join(context['dataset_overview']['columns'])}

        Data Type Breakdown:
        {json.dumps(context['data_types'], indent=2)}

        Describe the dataset's composition, highlighting unique characteristics,
        potential data quality issues, and the significance of each column type.
        """

        return self._send_llm_request(overview_prompt)

    def _generate_insights_prompt(self, context: Dict[str, Any]) -> str:

        # Generate insights prompt focusing on statistical patterns
        # Providing optimum amount of data
        # Context rich prompt
        insights_prompt = f"""
        Analyze statistical insights and patterns in the dataset:

        Feature Importance:
        {json.dumps(context['feature_importance'], indent=2, default=json_serialize_handler)}

        Normality Tests:
        {json.dumps(context['normality_tests'], indent=2, default=json_serialize_handler)}

        Outlier Analysis:
        {json.dumps(context['outliers'], indent=2, default=json_serialize_handler)}

        Clustering Insights:
        - Optimal Clusters: {context['clustering']['optimal_clusters']}

        Provide a deep dive into statistical relationships,
        unexpected patterns, potential correlations,
        and implications of the detected clusters and outliers.
        """

        return self._send_llm_request(insights_prompt)

    def _generate_recommendations_prompt(self, context: Dict[str, Any]) -> str:

        # Generate recommendations based on the analysis
        # Multiple LLM Prompts with reduced token usage
        recommendations_prompt = f"""
        Based on the comprehensive dataset analysis,
        provide strategic recommendations:

        Dataset Characteristics:
        - Total Rows: {context['dataset_overview']['total_rows']}
        - Columns: {', '.join(context['dataset_overview']['columns'])}

        Key Findings:
        - Clustering: Identified {context['clustering']['optimal_clusters']} clusters
        - Outliers Detected: {len(context['outliers'])} columns with outliers

        Recommend:
        1. Potential data preprocessing steps
        2. Strategies for handling identified outliers
        3. Insights for further investigation
        4. Potential machine learning approaches
        5. Business or research implications
        """

        return self._send_llm_request(recommendations_prompt)

    def generate_graph_suggestion(self, context: Dict[str, Any]) -> str:

        # Generate a graph suggestion from the LLM, implement it, and save the image file.
        # Agentic work flow
        # Prepare the context for graph suggestion
        context = {
            'dataset_overview': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'columns': list(self.df.columns)
            },
            'summary_statistics': self.df.describe().to_dict()
        }

        # Prepare the prompt for LLM
        prompt = f"""
        Dataset Overview:
        - Total Rows: {context['dataset_overview']['total_rows']}
        - Total Columns: {context['dataset_overview']['total_columns']}
        - Columns: {', '.join(context['dataset_overview']['columns'])}
        - File Name: {filename}
        Summary Statistics:
        {json.dumps(context['summary_statistics'], indent=2, default=json_serialize_handler)}

        Graphs already created: Cluster Analysis, Outlier Boxplot 

        Given the following dataset characteristics, suggest an additional graph or visualization
        to analyze the data effectively. Include Python code to implement the visualization
        using libraries like matplotlib or seaborn.
        Use Enhanced CSV loading with comprehensive encoding detection and validation
        encodings_to_try = [
            'utf-8', 'iso-8859-1', 'latin1', 'cp1252', 'utf-16',
            'ascii', 'big5', 'shift_jis', 'gb2312'
        ]
        The Python code should generate the visualization with dip=512/8, save it as 'llm_suggested_graph.png' and end with plt.close()
        Note: Give only the code with everything else commented inside the code itself.
        Please satisfy the condition in NOTE. Thank You!
        """
            
        # Send the prompt to the AI system
        ai_response = self._send_llm_request(prompt)
        # Parse the code generated by the LLM
        code = ai_response[9:len(ai_response)-4]
        try:
            # Execute the code generated by the LLM
            exec(code)
            return code
        except Exception as e:
            print(f"Error executing AI-suggested graph code: {e}")
            code = "LLM failed to give proper code for graph. So skip this part"
            return code


    def _compose_narrative(self, narrative_stages: List[str]) -> str:

        # Compose a coherent narrative from different analysis stages
        # Agentic work flow
        final_prompt = f"""
        Integrate the following analysis stages into a cohesive,
        storytelling narrative that provides a comprehensive
        understanding of the dataset:

        1. Dataset Overview
        {narrative_stages[0]}

        2. Statistical Insights
        {narrative_stages[1]}

        3. Strategic Recommendations
        {narrative_stages[2]}

        4. Code for Suggested Graph by the LLM
        {narrative_stages[3]}

        Create a narrative that:
        - Flows logically between sections
        - Highlights key discoveries
        - Provides actionable insights
        - Describe the LLM generated code (only if exists), Does not include the code itself
        - Maintains an engaging, professional tone
        """

        return self._send_llm_request(final_prompt)

    def _send_llm_request(self, prompt: str) -> str:

        # Send request to AI Proxy with robust error handling

        AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

        if not AIPROXY_TOKEN:
            raise ValueError("AIPROXY_TOKEN environment variable is not set")

        BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        # Optimum token usage
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        try:
            response = requests.post(BASE_URL, headers=headers, json=data)
            response.raise_for_status()

            # Extract and return narrative content
            narrative = response.json().get("choices")[0].get("message").get("content")
            return narrative

        except requests.exceptions.RequestException as e:
            return f"Error generating narrative: {e}"

    # Update the main method to use the summary generation
    def generate_comprehensive_report(self):

        # Generate a comprehensive report with LLM-powered narrative

        # analysis methods
        profiling = self.advanced_data_profiling()
        outliers = self.advanced_outlier_detection()
        clustering = self.advanced_clustering()

        # Generate LLM narrative
        llm_narrative = AdvancedDataAnalyzer.generate_llm_summary(self)

        # Create report
        report = f"""# Data Analysis Report

    {llm_narrative}

    """

        # Write to markdown
        with open('README.md', 'w') as f:
            f.write(report)
        return report

def main():

    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_filename>")
        sys.exit(1)
    global filename
    filename = sys.argv[1]
    analyzer = AdvancedDataAnalyzer(filename)
    analyzer.generate_comprehensive_report()
    with open('README.md', 'a') as f:
            f.write("\n\n## Visualizations\n")
            f.write("![Outliers Boxplot](outliers_boxplot.png)\n")
            f.write("![Cluster Analysis](cluster_analysis.png)\n")
            f.write("![LLM Suggested Graph](llm_suggested_graph.png)\n")
        

if __name__ == "__main__":
    main()
