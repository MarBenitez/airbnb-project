import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
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

def preprocess_features(df, target_column='price'):
    """
    Preprocess the DataFrame for model training.
    
    Parameters:
    - df: DataFrame containing the data.
    - target_column: The name of the column to predict.
    
    Returns:
    - X: Feature matrix.
    - y: Target vector.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # One-hot encode categorical variables (if any)
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def scale_features(X_train, X_test, sample_frac=0.1):
    """
    Scale the feature matrices using StandardScaler.
    
    Parameters:
    - X_train: Training feature matrix.
    - X_test: Testing feature matrix.
    
    Returns:
    - X_train_scaled: Scaled training feature matrix.
    - X_test_scaled: Scaled testing feature matrix.
    - scaler: Fitted scaler object.
    """
    scaler = StandardScaler()

    X_train_sampled = X_train.sample(frac=sample_frac, random_state=42)


    X_train_scaled = scaler.fit_transform(X_train_sampled)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    """
    Train a RandomForestRegressor model.
    
    Parameters:
    - X_train: Training feature matrix.
    - y_train: Training target vector.
    
    Returns:
    - model: Trained model.
    """
    model = RandomForestRegressor(random_state=42)
    
    # You can also perform hyperparameter tuning using GridSearchCV
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    # }
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X_train, y_train)
    # model = grid_search.best_estimator_

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.
    
    Parameters:
    - model: Trained model.
    - X_test: Testing feature matrix.
    - y_test: Testing target vector.
    
    Returns:
    - metrics: Dictionary containing evaluation metrics.
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

def save_model(model, scaler, model_path='models/price_prediction_model.pkl', scaler_path='models/scaler.pkl'):
    """
    Save the trained model and scaler to disk.
    
    Parameters:
    - model: Trained model.
    - scaler: Fitted scaler.
    - model_path: Path to save the model.
    - scaler_path: Path to save the scaler.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def main():
    # Load the data
    data_filepath = 'data/processed/engineered_listing.csv'
    df = load_data(data_filepath)
    
    # Preprocess the features and target
    X, y = preprocess_features(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test_scaled, y_test)
    print(f"Model evaluation metrics: {metrics}")
    
    # Save the model and scaler
    save_model(model, scaler)

if __name__ == '__main__':
    main()
