import pandas as pd
import numpy as np
from geopy.distance import great_circle
from sklearn.cluster import KMeans

def drop_inconsistent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are inconsistent or have many missing values.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with dropped columns.
    """
    # Drop columns that are inconsistent or have many missing values
    columns_to_drop = ['beds']
    df = df.drop(columns=columns_to_drop)
    
    return df

def calculate_distance_to_touristic_places(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distances from listings to key touristic places in Rio de Janeiro.

    Parameters:
        df (Pd.DataFrame): The DataFrame containing listing data.

    Returns:
        pd.DataFrame: The DataFrame with new columns for distances to touristic places.
    """
    # Define the coordinates of key touristic places
    touristic_places = {
        'cristo_redentor': (-22.9519, -43.2105),
        'pan_de_azucar': (-22.9486, -43.1553),
        'copacabana_beach': (-22.9711, -43.1822),
        'ipanema_beach': (-22.9839, -43.2045),
        'botanical_garden': (-22.9674, -43.2292)
    }
    
    # Perform clustering by region
    num_clusters = 156  # Number of neighborhoods
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['region'] = kmeans.fit_predict(df[['latitude', 'longitude']])
    
    # Calculate distances to each touristic place
    for place, coords in touristic_places.items():
        distance_col = f'distance_to_{place}'
        df[distance_col] = df.apply(lambda row: great_circle((row['latitude'], row['longitude']), coords).kilometers, axis=1)
        df[distance_col] = pd.to_numeric(df[distance_col], errors='coerce')
    
    return df

def perform_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all feature engineering steps.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """
    df = drop_inconsistent_columns(df)
    df = calculate_distance_to_touristic_places(df)
    
    return df
