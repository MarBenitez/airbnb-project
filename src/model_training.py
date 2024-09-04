import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json


def load_data(filepath):
    """
    Load the preprocessed data from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_features(df, target_column='price'):
    """
    Preprocess the DataFrame for model training.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # One-hot encode categorical variables (if any)
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def scale_features(X_train, X_test):
    """
    Scale the feature matrices using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    """
    Train a RandomForestRegressor model.
    """
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    
    return metrics

def save_metrics(metrics, filepath='models/metrics-rf.json'):
    """
    Save the evaluation metrics to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {filepath}")

def save_model(model, scaler, model_path='models/model-rf.pkl', scaler_path='models/scaler-rf.pkl'):
    """
    Save the trained model and scaler to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def save_trained_columns(X, filepath='models/trained_columns.pkl'):
    """
    Save the column names used during training (after preprocessing).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    trained_columns = X.columns.tolist()
    joblib.dump(trained_columns, filepath)
    print(f"Trained columns saved to {filepath}")

def main():
    # Load the data
    data_filepath = 'C:/Users/mar27/OneDrive/Documentos/GitHub/airbnb-project/data/testing/train_data.csv'
    df = load_data(data_filepath)
    
    # Preprocess the features and target
    X, y = preprocess_features(df)
    
    # Save the column names used during training
    save_trained_columns(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test_scaled, y_test)
    print(f"Model evaluation metrics: {metrics}")
    
    # Save the model, scaler, and metrics
    save_model(model, scaler)
    save_metrics(metrics)

if __name__ == '__main__':
    main()
