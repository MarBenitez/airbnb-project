import pandas as pd
import geopandas as gpd
from typing import Dict

def load_data(filepaths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load datasets from a dictionary of file paths.

    Parameters:
        filepaths (dict): A dictionary with file names as keys and file paths as values.

    Returns:
        dict: A dictionary with file names as keys and DataFrames as values.
    """
    data = {}
    for name, path in filepaths.items():
        if path.endswith('.geojson'):
            data[name] = gpd.read_file(path)
        else:
            data[name] = pd.read_csv(path)
    return data

def clean_listing_data(listing: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the listing DataFrame by dropping unnecessary columns and converting data types.

    Parameters:
        listing (pd.DataFrame): The raw listing DataFrame.

    Returns:
        pd.DataFrame: The cleaned listing DataFrame.
    """
    listing = listing.drop(['license', 'neighbourhood_group', 'neighbourhood'], axis=1)
    listing['last_review'] = pd.to_datetime(listing['last_review'])
    return listing

def clean_listing_details(listing_details: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the listing_details DataFrame by dropping empty columns.

    Parameters:
        listing_details (pd.DataFrame): The raw listing_details DataFrame.

    Returns:
        pd.DataFrame: The cleaned listing_details DataFrame.
    """
    empty_columns_ld = ['description', 'bathrooms', 'license', 'calendar_updated', 'bedrooms', 'neighbourhood_group_cleansed']
    listing_details = listing_details.drop(empty_columns_ld, axis=1)
    return listing_details

def merge_listings_with_details(listing: pd.DataFrame, listing_details: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the cleaned listing DataFrame with specific columns from listing_details.

    Parameters:
        listing (pd.DataFrame): The cleaned listing DataFrame.
        listing_details (pd.DataFrame): The cleaned listing_details DataFrame.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    target_columns = [
        'id', 'neighbourhood_cleansed', 'host_response_time', 
        'host_response_rate', 'host_is_superhost', 'host_listings_count', 
        'host_identity_verified', 'accommodates', 'beds', 'review_scores_rating', 
        "review_scores_cleanliness", 'review_scores_location', 'review_scores_accuracy', 
        'review_scores_communication', 'review_scores_checkin',  'review_scores_value', 
        'property_type', 'host_acceptance_rate', 'maximum_nights', 'listing_url'
    ]
    merged_df = pd.merge(listing, listing_details[target_columns], on='id', how='left')
    merged_df['host_response_rate'] = pd.to_numeric(merged_df['host_response_rate'].str.strip('%'))
    return merged_df

def clean_duplicated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated rows from the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame without duplicated rows.
    """
    df = df.drop_duplicates()
    return df

def preprocess_data(filepaths: Dict[str, str], save_merged: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load, clean, and preprocess all necessary data.

    Parameters:
        filepaths (dict): A dictionary with file names as keys and file paths as values.
        save_merged (bool): If True, save the merged listing DataFrame to a CSV file.

    Returns:
        dict: A dictionary with preprocessed DataFrames.
    """
    data = load_data(filepaths)
    
    data['listing'] = clean_listing_data(data['listing'])
    data['listing_details'] = clean_listing_details(data['listing_details'])
    
    merged_listing = merge_listings_with_details(data['listing'], data['listing_details'])
    merged_listing = clean_duplicated_rows(merged_listing)
    
    if save_merged:
        merged_listing.to_csv('data/intermediate/merged_listing.csv', index=False)
    
    data['merged_listing'] = merged_listing
    
    return data
