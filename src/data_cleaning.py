import pandas as pd
import numpy as np
import logging
from src.visualization import plot_boxplot, plot_histogram
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_zero_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the price is zero.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    logging.info(f"Removing rows where price is zero. Initial number of rows: {df.shape[0]}")
    df_cleaned = df[df['price'] != 0]
    logging.info(f"Rows after removing zero price: {df_cleaned.shape[0]}")
    return df_cleaned

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    df['price'] = df.groupby(['neighbourhood_cleansed', 'room_type'])['price'].transform(lambda x: x.fillna(x.mean()))
    df['price'] = df['price'].fillna(df['price'].median())
    df = df.dropna(subset=['last_review'])
    df.loc[:, 'host_response_rate'] = df['host_response_rate'].fillna(df['host_response_rate'].mean())
    df.loc[:, 'host_response_time'] = df['host_response_time'].fillna(df['host_response_time'].mode()[0])
    df.loc[:, 'host_is_superhost'] = df['host_is_superhost'].fillna(df['host_is_superhost'].mode()[0])
    df.loc[:, 'host_acceptance_rate'] = df.groupby('host_response_rate')['host_acceptance_rate'].transform(lambda x: x.fillna(x.median()))
    
    review_columns = [
        'review_scores_rating', 'review_scores_cleanliness', 'review_scores_location',
        'review_scores_accuracy', 'review_scores_communication', 'review_scores_checkin',
        'review_scores_value'
    ]
    df = df.dropna(subset=review_columns)
    
    logging.info("Missing values handled successfully.")
    return df

def handle_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Handle outliers in a column using either the IQR method or Z-Score method.

    - IQR Method: Replaces outliers with the upper and lower bounds.
    - Z-Score Method: Replaces values beyond 3 standard deviations with the upper and lower bounds.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        column (str): The column where outliers will be identified and handled.
        method (str): Method to handle outliers, either 'iqr' or 'zscore'.

    Returns:
        pd.DataFrame: The DataFrame with outliers handled.
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

    # Replace values outside bounds with the respective bounds
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])

    logging.info(f"Outliers handled using {method} method for column: {column}.")
    return df


def create_folder_if_not_exists(path):
    """
    Create the folder if it doesn't exist.
    
    Parameters:
        path (str): The directory path to check and create if missing.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created folder: {path}")
    else:
        logging.info(f"Folder already exists: {path}")

def clean_data(df: pd.DataFrame, vis_folder: str = 'visualizations/cleaned_data') -> pd.DataFrame:
    """
    Perform the complete data cleaning process:
    - Handle missing values.
    - Remove rows where price is zero.
    - Handle outliers using the IQR or Z-score method.
    - Generate and save plots after each cleaning step.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        vis_folder (str): Folder where visualizations will be saved.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    logging.info("Starting the data cleaning process...")
    
    # Ensure the folder for visualizations exists
    create_folder_if_not_exists(vis_folder)

    # Handling missing values
    df_cleaned = handle_missing_values(df)

    plot_histogram(df_cleaned, 'price', title="Price Distribution After Missing Values Handling",
                   save_path=os.path.join(vis_folder, "price_dist_after_missing.png"))
    plot_boxplot(df_cleaned, 'price', title="Price Boxplot After Missing Values Handling",
                 save_path=os.path.join(vis_folder, "price_boxplot_after_missing.png"))

    df_cleaned = remove_zero_price(df_cleaned)

    plot_histogram(df_cleaned, 'price', title="Price Distribution After Removing Zero Price",
                   save_path=os.path.join(vis_folder, "price_dist_after_zero_price.png"))
    plot_boxplot(df_cleaned, 'price', title="Price Boxplot After Removing Zero Price",
                 save_path=os.path.join(vis_folder, "price_boxplot_after_zero_price.png"))

    df_cleaned = handle_outliers(df_cleaned, 'price', method='iqr')

    plot_histogram(df_cleaned, 'price', title="Price Distribution After Outlier Handling",
                   save_path=os.path.join(vis_folder, "price_dist_after_outliers.png"))
    plot_boxplot(df_cleaned, 'price', title="Price Boxplot After Outlier Handling",
                 save_path=os.path.join(vis_folder, "price_boxplot_after_outliers.png"))

    logging.info("Data cleaning process completed.")
    
    return df_cleaned
