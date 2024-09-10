import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import xgboost as xgb

def load_data(filepath):
    """
    Load data from a CSV file.
    
    Parameters:
    - filepath: Path to the CSV file containing the dataset.
    
    Returns:
    - df: Loaded DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def split_data(df, target='price', test_size=0.2, random_state=42):
    """
    Split the dataset into training and test sets.
    
    Parameters:
    - df: DataFrame containing the data.
    - target: Name of the target column (default 'price').
    - test_size: Proportion of the data to be used for testing.
    - random_state: Seed used by the random number generator.
    
    Returns:
    - X_train, X_test, y_train, y_test: Split data.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, model):
    """
    Train the machine learning model.
    
    Parameters:
    - X_train: Training features.
    - y_train: Target values for training.
    - model: Machine learning model instance.
    
    Returns:
    - Trained model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the model using test data.
    
    Parameters:
    - model: Trained machine learning model.
    - X_test: Test features.
    - y_test: True target values for the test set.
    
    Returns:
    - mse: Mean Squared Error.
    - rmse: Root Mean Squared Error.
    - r2: R-squared score.
    - mae: Mean Absolute Error.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, r2, mae

def hyperparameter_tuning(X_train, y_train, model, param_dist):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Parameters:
    - X_train: Training features.
    - y_train: Target values for training.
    - model: Machine learning model instance.
    - param_dist: Dictionary containing hyperparameter search space.
    
    Returns:
    - best_estimator: Model with the best hyperparameters.
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

def save_model(model, filename):
    """
    Save the trained model to a file.
    
    Parameters:
    - model: Trained machine learning model.
    - filename: Path where the model will be saved.
    """
    joblib.dump(model, filename)

def main():
    """
    Main function to load data, preprocess, perform model training, hyperparameter tuning, and evaluation.
    """
    data_filepath = './data/modeling/modeling_data.csv'
    df = load_data(data_filepath)
    df = df.drop(columns=df.select_dtypes(include='object').columns)
    
    X_train, X_test, y_train, y_test = split_data(df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': [int(x) for x in np.linspace(200, 2000, 10)],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [int(x) for x in np.linspace(10, 110, 11)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    best_rf = hyperparameter_tuning(X_train_scaled, y_train, rf, param_dist)
    mse, rmse, r2, mae = evaluate_model(best_rf, X_test_scaled, y_test)
    print(f'Best RF: MSE = {mse}, RMSE = {rmse}, R2 = {r2}, MAE = {mae}')
    
    save_model(best_rf, './models/module/best_rf_model.pkl')

    xgb_model = xgb.XGBRegressor(random_state=42)
    param_dist_xgb = {
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'min_child_weight': [1, 2, 3, 4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0, 0.01, 0.1, 1]
    }
    
    best_xgb = hyperparameter_tuning(X_train_scaled, y_train, xgb_model, param_dist_xgb)
    mse, rmse, r2, mae = evaluate_model(best_xgb, X_test_scaled, y_test)
    print(f'Best XGB: MSE = {mse}, RMSE = {rmse}, R2 = {r2}, MAE = {mae}')
    
    save_model(best_xgb, './models/module/best_xgb_model.pkl')

if __name__ == '__main__':
    main()
