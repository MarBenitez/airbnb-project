import pandas as pd
import joblib
import os

def load_inference_data(filepath):
    """
    Load the inference data from a CSV file.
    
    Parameters:
    - filepath: Path to the CSV file.
    
    Returns:
    - df: Loaded DataFrame for inference.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_features_for_inference(df, trained_columns):
    """
    Preprocess the DataFrame for inference, ensuring it matches the training data structure.
    
    Parameters:
    - df: DataFrame containing the inference data.
    - trained_columns: List of columns used during training.
    
    Returns:
    - X: Preprocessed feature matrix for inference.
    """
    # One-hot encode categorical variables
    X = pd.get_dummies(df, drop_first=True)
    
    # Align columns with training data (add missing columns and reorder)
    X = X.reindex(columns=trained_columns, fill_value=0)
    
    return X

def scale_features_for_inference(X, scaler):
    """
    Scale the feature matrix using the saved scaler.
    
    Parameters:
    - X: Feature matrix for inference.
    - scaler: Fitted scaler object.
    
    Returns:
    - X_scaled: Scaled feature matrix for inference.
    """
    X_scaled = scaler.transform(X)
    return X_scaled

def main():
    # Load the inference data
    inference_data_filepath = 'C:/Users/mar27/OneDrive/Documentos/GitHub/airbnb-project/data/testing/inference_data.csv'
    df_inference = load_inference_data(inference_data_filepath)
    
    # Load the trained model and scaler
    model = joblib.load('models/model-rf.pkl')
    scaler = joblib.load('models/scaler-rf.pkl')
    
    # Load the trained columns (saved during training)
    trained_columns_path = 'models/trained_columns.pkl'
    if os.path.exists(trained_columns_path):
        with open(trained_columns_path, 'rb') as f:
            trained_columns = joblib.load(f)
    else:
        raise FileNotFoundError("Trained columns file not found. Make sure to save the columns during training.")
    
    # Preprocess the features for inference, ensuring they match the training columns
    X_inference = preprocess_features_for_inference(df_inference, trained_columns)
    
    # Scale the features using the same scaler as during training
    X_inference_scaled = scale_features_for_inference(X_inference, scaler)
    
    # Make predictions
    predictions = model.predict(X_inference_scaled)
    
    # Save predictions to a CSV file
    df_inference['predicted_price'] = predictions
    output_filepath = 'data/testing/predicted_data-rf.csv'
    df_inference.to_csv(output_filepath, index=False)
    
    print(f"Predictions saved to {output_filepath}")

if __name__ == '__main__':
    main()
