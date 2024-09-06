import os
import pandas as pd
import logging
from src.visualization import plot_boxplot, plot_price_by_neighbourhood, plot_histogram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_missing_values(df: pd.DataFrame, suffix: str, output_path: str = 'data/intermediate/') -> pd.Series:
    """
    Analyze missing values in the DataFrame and save the result in CSV.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        suffix (str): Suffix to append to the output file name.
        output_path (str): Path where the CSV file will be saved.

    Returns:
        pd.Series: A series containing the count of missing values per column, sorted by the count.
    """
    missing_values = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
    
    output_file = os.path.join(output_path, f'missing_values_{suffix}.csv')
    missing_values.to_csv(output_file, header=['missing_count'])
    
    logging.info(f"Missing values saved to {output_file}")
    
    return missing_values


def basic_info(df: pd.DataFrame, suffix: str, output_path: str = 'data/intermediate/', vis_folder: str = 'visualizations/'):
    """
    Display basic information about the DataFrame, log the data, and call plot functions.

    Parameters:
        df (pd.DataFrame): The DataFrame to display information about.
        suffix (str): Suffix to append to file names to avoid overwriting.
        output_path (str): Directory where CSV files will be saved.
        vis_folder (str): Folder where visualizations will be saved.
    """
    logging.info(f"Analyzing DataFrame with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    analyze_missing_values(df, suffix=suffix, output_path=output_path)
    
    df_info = df.describe()
    basic_info_file = os.path.join(output_path, f'basic_info_{suffix}.csv')
    df_info.to_csv(basic_info_file)
    logging.info(f"Basic statistics saved to {basic_info_file}")
    
    vis_path = os.path.join(vis_folder, suffix)
    os.makedirs(vis_path, exist_ok=True)
    
    plot_price_by_neighbourhood(df, save_path=os.path.join(vis_path, f"price_by_neighbourhood_{suffix}.png"))
    plot_histogram(df, 'price', title="Price Distribution", save_path=os.path.join(vis_path, f"price_distribution_{suffix}.png"))
    plot_boxplot(df, 'price', title="Price Boxplot", save_path=os.path.join(vis_path, f"price_boxplot_{suffix}.png"))
    
    logging.info(f"Plots saved in {vis_path}")
