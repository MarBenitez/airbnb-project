import os
import pandas as pd
from scipy.stats import shapiro, kstest
from sklearn.preprocessing import LabelEncoder
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def encode_categorical_columns(df, categorical_columns):
    """
    Encode categorical variables using LabelEncoder.

    Parameters:
    - df: DataFrame containing the data
    - categorical_columns: List of columns to encode

    Returns:
    - df_encoded: DataFrame with encoded categorical columns
    """
    df_encoded = df.copy()
    le = LabelEncoder()

    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded

def create_folder_if_not_exists(path):
    """
    Create the folder if it doesn't exist.

    Parameters:
    - path (str): Directory path to create if missing.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created folder: {path}")
    else:
        logging.info(f"Folder already exists: {path}")

def test_normality(df, columns=None, save_path=None, file_suffix=''):
    """
    Test for normality using Shapiro-Wilk and Kolmogorov-Smirnov tests.
    
    Parameters:
    - df: DataFrame containing the data
    - columns: List of columns to test. If None, tests all numeric columns.
    - save_path: Path to save the JSON result. Will be saved in 'results/normality_tests'.
    - file_suffix: Suffix to distinguish between different stages of normality testing (e.g., after cleaning).
    
    Returns:
    - normality_results: Dictionary with results for each column
    """
    normality_results = {}
    
    # If columns are not provided, default to all numeric columns
    if columns is None:
        columns = df.select_dtypes(include='number').columns
    
    for column in columns:
        data = df[column].dropna()
        shapiro_stat, shapiro_p = shapiro(data)
        ks_stat, ks_p = kstest(data, 'norm')
        
        normality_results[column] = {
            'Shapiro-Wilk': {'Statistic': shapiro_stat, 'p-value': shapiro_p},
            'Kolmogorov-Smirnov': {'Statistic': ks_stat, 'p-value': ks_p},
            'is_normal': str(shapiro_p > 0.05 and ks_p > 0.05)  # Convert boolean to string for JSON compatibility
        }
    
    # Ensure the save_path exists
    create_folder_if_not_exists(save_path)
    
    # Construct the output file name with suffix
    output_file = os.path.join(save_path, f'normality_test_{file_suffix}.json')

    # Write the results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(normality_results, f, indent=4)
    
    logging.info(f"Normality test results saved to {output_file}")
    
    return normality_results

def calculate_correlation_matrix(df, method='spearman', save_path=None):
    """
    Calculate the correlation matrix using a specified method.

    Parameters:
    - df: DataFrame containing the data
    - method: Method of correlation ('pearson', 'spearman', 'kendall')

    Returns:
    - corr_matrix: Correlation matrix
    """
    numeric_columns = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_columns].corr(method=method)

    return corr_matrix

def find_significant_correlations(corr_matrix, threshold=0.7, save_path=None):
    """
    Find pairs of variables with correlation higher than a threshold.

    Parameters:
    - corr_matrix: Correlation matrix
    - threshold: Threshold for significant correlation

    Returns:
    - correlated_pairs_df: DataFrame with significant correlations
    """
    correlated_pairs = []

    for col in corr_matrix.columns:
        for row in corr_matrix.index:
            if col != row:  
                correlation_value = corr_matrix.loc[row, col]
                if abs(correlation_value) > threshold:
                    correlated_pairs.append((row, col, correlation_value))

    correlated_pairs_df = pd.DataFrame(correlated_pairs, columns=['Variable 1', 'Variable 2', 'Correlation'])
    correlated_pairs_df = correlated_pairs_df.drop_duplicates(subset=['Correlation'])

    return correlated_pairs_df
