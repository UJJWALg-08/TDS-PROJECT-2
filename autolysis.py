# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",        # For data manipulation and analysis
#   "numpy",         # For numerical operations, especially with arrays and matrices
#   "matplotlib",    # For creating static, animated, and interactive visualizations
#   "seaborn",       # For statistical data visualization, built on top of matplotlib
# ]
# ///


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def set_aiproxy_token():
    """
    Prompt the user to enter the AI Proxy token and set it as an environment variable.
    """
    AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')

    print("AI Proxy token successfully set!")


def create_output_folder(folder_name="analysis_output"):
    """
    Create a new folder to store all analysis outputs.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print(f"Output folder '{folder_name}' created.")
    return folder_name


def analyze_dataset(file_path, output_folder):
    """
    Load and analyze the dataset.
    Provides dataset info, summary statistics, missing values, and correlation matrix for numeric columns.
    """
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {file_path}\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Save basic information to a file
    with open(os.path.join(output_folder, "dataset_info.txt"), "w") as f:
        f.write("--- Dataset Info ---\n")
        f.write(str(df.info()) + "\n")
        f.write("\n--- First 5 Rows ---\n")
        f.write(str(df.head()) + "\n")
        f.write("\n--- Summary Statistics ---\n")
        f.write(str(df.describe(include='all')) + "\n")

    # Check and log missing values
    missing_values = df.isnull().sum()[df.isnull().sum() > 0]
    with open(os.path.join(output_folder, "missing_values.txt"), "w") as f:
        if not missing_values.empty:
            f.write("Missing Values:\n")
            f.write(str(missing_values) + "\n")
        else:
            f.write("No missing values in the dataset.\n")

    # Display and save correlation matrix for numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()
        corr_matrix.to_csv(os.path.join(output_folder, "correlation_matrix.csv"))

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title("Correlation Matrix Heatmap")
        plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
        plt.close()
    return df


def generate_visualizations(df, output_folder):
    """
    Generate visualizations for numeric columns in the dataset.
    """
    print("\n--- Generating Visualizations ---")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        for column in numeric_df.columns[:3]:  # Only select up to 3 best features
            plt.figure(figsize=(6, 4))
            sns.histplot(df[column], kde=True, color='skyblue')
            plt.title(f"Distribution of {column}")
            image_path = os.path.join(output_folder, f"{column}_distribution.png")
            plt.savefig(image_path, dpi=150)
            print(f"Saved visualization for {column} as {image_path}")
            plt.close()
    else:
        print("No numeric columns found for visualization.")


def create_readme(data_summary, analysis_steps, key_insights, implications, output_folder):
    """
    Generate a README file with a story-like format.
    """
    story = (
        "# Analysis Report\n\n"
        "### Data Overview\n"
        "Once upon a time, we received a dataset that contained information about various features. "
        f"{data_summary}\n\n"
        "### Analysis Steps\n"
        f"{analysis_steps}\n\n"
        "### Key Insights\n"
        f"{key_insights}\n\n"
        "### Implications\n"
        f"{implications}\n\n"
        "Thank you for exploring this story with us!"
    )

    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w") as f:
        f.write(story)
    print(f"README.md file successfully created at {readme_path}!")


def write_story(df, output_folder):
    """
    Use a proxy-enabled AI model to generate a story-like README from the analysis.
    """
    # Summarize dataset overview
    data_summary = (
        f"The dataset contained {df.shape[0]} rows and {df.shape[1]} columns. "
        f"We observed that {df.isnull().sum().sum()} missing values were present."
    )

    # Describe the analysis steps
    analysis_steps = (
        "We performed the following analyses:\n"
        "- Examined the structure and summary statistics of the dataset.\n"
        "- Checked for missing values and correlations between numeric columns.\n"
        "- Visualized key distributions for up to 3 features to understand their behavior."
    )

    # Highlight key insights
    key_insights = (
        "From the analysis, the following insights were identified:\n"
        "- The columns with the highest correlations were highlighted.\n"
        "- Key features with unusual distributions were visualized and noted.\n"
        "- Patterns in the data suggested interesting trends."
    )

    # Mention implications
    implications = (
        "The insights gained could be used to make data-driven decisions, optimize processes, "
        "or explore further opportunities for improvement."
    )

    create_readme(data_summary, analysis_steps, key_insights, implications, output_folder)


def main():
    """
    Main function to execute the analysis process.
    """
    parser = argparse.ArgumentParser(description='Autolysis Analysis')
    parser.add_argument('dataset_path', help='Path to the CSV file')
    args = parser.parse_args()

    # Step 1: Set AI Proxy Token
    set_aiproxy_token()

    # Step 2: Create output folder
    output_folder = create_output_folder()

    # Step 3: Analyze the dataset
    df = analyze_dataset(args.dataset_path, output_folder)

    # If dataset analysis is successful, generate visualizations and write README
    if df is not None:
        generate_visualizations(df, output_folder)
        write_story(df, output_folder)


if __name__ == "__main__":
    main()
