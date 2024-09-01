import pandas as pd
import numpy as np
from scipy.stats import kstest
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

def test_normality2(df, columns):
    """
    Test for normality using Kolmogorov-Smirnov test.
    
    Parameters:
    - df: DataFrame containing the data
    - columns: List of columns to test
    
    Returns:
    - normality_results: Dictionary with results for each column
    """
    normality_results = {}
    
    for column in columns:
        data = df[column].dropna()
        stat, p_value = kstest(data, 'norm')
        
        normality_results[column] = {
            'Statistic': stat,
            'p-value': p_value,
            'is_normal': p_value > 0.05
        }
        
        print(f"Column: {column}, Statistic={stat}, p-value={p_value}")
        if p_value > 0.05:
            print(f"The column '{column}' appears to be normally distributed (fail to reject H0).")
        else:
            print(f"The column '{column}' does NOT appear to be normally distributed (reject H0).")
    
    return normality_results

def calculate_correlation_matrix(df, method='spearman'):
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

def plot_correlation_matrix(corr_matrix):
    """
    Plot the correlation matrix using a heatmap.
    
    Parameters:
    - corr_matrix: Correlation matrix to plot
    """
    plt.figure(figsize=(10, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt=".2f")
    plt.show()

def plot_correlation_heatmap(corr_matrix):
    """
    Plot the correlation matrix using Plotly.
    
    Parameters:
    - corr_matrix: Correlation matrix to plot
    """
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    masked_corr_matrix = corr_matrix.mask(mask)

    fig = go.Figure(data=go.Heatmap(
        z=masked_corr_matrix.values,
        x=masked_corr_matrix.columns,
        y=masked_corr_matrix.index,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        text=masked_corr_matrix.values,
        hoverinfo='text'
    ))

    fig.update_layout(
        title='Correlation Matrix (Spearman)',
        xaxis_nticks=len(corr_matrix.columns),
        yaxis_nticks=len(corr_matrix.index),
        width=1000,
        height=1000
    )
    
    fig.show()

def find_significant_correlations(corr_matrix, threshold=0.7):
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
