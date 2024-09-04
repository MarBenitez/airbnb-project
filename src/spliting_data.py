# This module will be responsible for splitting the clean dataset I currently have into two main parts (in testing/ folder):

# 1. Training Data: This portion will be used to train the machine learning models. This training data will be further divided into train and test sets (val optional). 
# The train set will be used to fit the model, while the test set will be used to evaluate its performance and generalization capabilities.

# 2. New Data for Inference: The second part will be treated as if it's new, unseen data for making predictions (inference). 
# These data will have the target (Y) removed, and I will not verify if the predictions are correct or not. 
# These will be considered as data for which we don't have the target, simulating real-world new input where we want to predict the target (Y) based on the model.

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    """
    Load the preprocessed data from a CSV file.
    
    Parameters:
    - filepath: Path to the CSV file.
    
    Returns:
    - df: Loaded DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def split_data(df, target_column='price', test_size=0.8, random_state=42):
    """
    Split the dataset into two parts: one for training and one for inference (new data).
    
    Parameters:
    - df: DataFrame containing the cleaned data.
    - target_column: The name of the target column.
    - test_size: Proportion of the data to use for inference.
    - random_state: Random seed for reproducibility.
    
    Returns:
    - train_df: DataFrame for training (with target).
    - inference_df: DataFrame for inference (target removed).
    """
    # Split the data into training and inference parts
    train_df, inference_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Remove the target column from the inference dataset to simulate new data
    inference_df = inference_df.drop(columns=[target_column])
    
    return train_df, inference_df

def save_data(train_df, inference_df, output_dir='data/testing/'):
    """
    Save the train and inference datasets to CSV files in the specified directory.
    
    Parameters:
    - train_df: DataFrame for training.
    - inference_df: DataFrame for inference.
    - output_dir: Directory where the datasets will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_data.csv')
    inference_path = os.path.join(output_dir, 'inference_data.csv')
    
    train_df.to_csv(train_path, index=False)
    inference_df.to_csv(inference_path, index=False)
    
    print(f"Train data saved to: {train_path}")
    print(f"Inference data saved to: {inference_path}")

def main():
    # Load the clean data from the processed file
    data_filepath = 'C:/Users/mar27/OneDrive/Documentos/GitHub/airbnb-project/data/processed/engineered_listing.csv'
    df = load_data(data_filepath)
    
    # Split the data into training data and inference data
    train_df, inference_df = split_data(df)
    
    # Save the split data into the testing/ folder
    save_data(train_df, inference_df)

if __name__ == '__main__':
    main()
