import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import zscore, shapiro, kstest

def remove_zero_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the price is zero.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    return df[df['price'] != 0]

def identify_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Identify outliers using the IQR method.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        column (str): The column name to check for outliers.

    Returns:
        pd.DataFrame: DataFrame containing outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def identify_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
    """
    Identify outliers using the Z-score method.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        column (str): The column name to check for outliers.
        threshold (float): The Z-score threshold to define outliers.

    Returns:
        pd.DataFrame: DataFrame containing outliers.
    """
    df_no_na = df.dropna(subset=[column])
    z_scores = zscore(df_no_na[column])
    abs_z_scores = np.abs(z_scores)
    return df_no_na[abs_z_scores > threshold]

def handle_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Handle outliers in a column using the specified method.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        column (str): The column name to clean.
        method (str): The method to use ('iqr' or 'zscore').

    Returns:
        pd.DataFrame: The DataFrame with outliers handled (outliers transformed to bounds).
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'zscore':
        mean_col = df[column].mean()
        std_col = df[column].std()
        lower_bound = mean_col - 3 * std_col
        upper_bound = mean_col + 3 * std_col
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df

def remove_outliers(df: pd.DataFrame, columns: list, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers from the DataFrame using the specified method.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        columns (list): List of columns to clean.
        method (str): The method to use ('iqr' or 'zscore').

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    for column in columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        elif method == 'zscore':
            z_scores = zscore(df[column].dropna())
            df = df[(np.abs(z_scores) <= 3)]
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    df = df.drop(columns='price_R', errors='ignore')
    df['price'] = df.groupby(['neighbourhood_cleansed', 'room_type'])['price'].transform(lambda x: x.fillna(x.mean()))
    df['price'] = df['price'].fillna(df['price'].median())
    df = df.dropna(subset=['last_review'])
    df['host_response_rate'] = df['host_response_rate'].fillna(df['host_response_rate'].mean())
    df['host_response_time'] = df['host_response_time'].fillna(df['host_response_time'].mode()[0])
    df['host_is_superhost'] = df['host_is_superhost'].fillna(df['host_is_superhost'].mode()[0])
    df['host_acceptance_rate'] = df.groupby('host_response_rate')['host_acceptance_rate'].transform(lambda x: x.fillna(x.median()))
    review_columns = [
        'review_scores_rating', 'review_scores_cleanliness', 'review_scores_location',
        'review_scores_accuracy', 'review_scores_communication', 'review_scores_checkin',
        'review_scores_value'
    ]
    df = df.dropna(subset=review_columns)
    return df