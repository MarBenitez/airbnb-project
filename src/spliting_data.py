import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the preprocessed and encoded dataset from a CSV file.

    Parameters:
    - filepath: str. Path to the CSV file containing the dataset.

    Returns:
    - df: pandas DataFrame. Loaded dataset.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the file exists and the path is correct.")
        raise

def split_data(df, target_column='price', test_size=0.2, random_state=42):
    """
    Split the dataset into two parts: one for modeling and one for inference (new data - real-world simulation).

    Parameters:
    - df: pandas DataFrame. The dataset to be split.
    - target_column: str. The name of the target column (e.g., 'price').
    - test_size: float. Proportion of the data to reserve for inference (default is 0.2).
    - random_state: int. Random seed for reproducibility.

    Returns:
    - modeling_df: pandas DataFrame. Modeling dataset including the target (for train and test later on).
    - inference_df: pandas DataFrame. Inference dataset with the target column removed.
    """
    modeling_df, inference_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    inference_df = inference_df.drop(columns=[target_column])
    
    return modeling_df, inference_df

def save_data(train_df, inference_df, output_dir='data/testing/'):
    """
    Save the modeling and inference datasets to CSV files.

    Parameters:
    - modeling_df: pandas DataFrame. Modeling dataset including the target column (for train and test later on).
    - inference_df: pandas DataFrame. Inference dataset with the target column removed.
    - output_dir: str. Directory where the datasets will be saved (default is 'data/modeling/').

    Returns:
    None.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    modeling_path = os.path.join(output_dir, 'modeling_data.csv')
    inference_path = os.path.join(output_dir, 'inference_data.csv')
    
    train_df.to_csv(modeling_path, index=False)
    inference_df.to_csv(inference_path, index=False)
    
    logging.info(f"Modeling data saved to: {modeling_path}")
    logging.info(f"Inference data saved to: {inference_path}")

def main():
    """
    Main function to execute the data loading, splitting, and saving process.
    
    Steps:
    - Load the dataset from the processed directory.
    - Split the data into training and inference parts.
    - Save the split datasets into the 'data/testing/' directory.
    
    Returns:
    None.
    """
    data_filepath = 'data/processed/engineered_listing_encoded.csv'
    
    engineered_listing_encoded = load_data(data_filepath)
    
    modeling_df, inference_df = split_data(engineered_listing_encoded, target_column='price', test_size=0.2, random_state=42)
    
    save_data(modeling_df, inference_df, output_dir='data/modeling/')

if __name__ == '__main__':
    main()
