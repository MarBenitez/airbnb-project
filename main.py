import warnings
import logging
import os
from src.data_loading import load_data
from src.data_preprocessing import load_preprocess_data
from src.EDA import basic_info
from src.visualization import (create_tourist_map, 
    plot_correlation_matrix
)
from src.data_cleaning import clean_data
from src.feature_engineering import perform_feature_engineering
from src.correlations import (
    encode_categorical_columns, 
    test_normality, 
    calculate_correlation_matrix,  
    find_significant_correlations
)

warnings.filterwarnings("ignore", category=UserWarning, module='scipy.stats')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

filepaths = {
    'listing': 'data/raw/listings.csv',
    'listing_details': 'data/raw/listings.csv.gz',
    'reviews': 'data/raw/reviews.csv',
    'reviews_details': 'data/raw/reviews.csv.gz',
    'neighbourhoods': 'data/raw/neighbourhoods.csv',
    'neighbourhoods_geo': 'data/raw/neighbourhoods.geojson',
    'calendar': 'data/raw/calendar.csv.gz'
}

def main():
    data = load_data(filepaths)

    data_preprocessed = load_preprocess_data(data, save_merged=True)
    logging.info("Data preprocessing completed, file saved in 'data/intermediate/'.")
    
    df_preprocessed = data_preprocessed['merged_listing']
    
    basic_info(df_preprocessed, suffix='preprocessed')
    logging.info("Basic EDA for preprocessed data completed, missing values saved to 'data/intermediate/'.")
    
    df_cleaned = clean_data(df_preprocessed)

    logging.info('Performing normality tests after handling outliers...')
    numeric_vars = df_cleaned.select_dtypes(include='number').columns
    
    test_normality(df_cleaned, columns=numeric_vars, save_path='results/normality_tests', file_suffix='after_cleaning')
    logging.info("Normality tests after cleaning saved to 'results/normality_tests'.")

    df_engineered = perform_feature_engineering(df_cleaned)
    df_engineered.to_csv('data/processed/engineered_listing.csv', index=False)
    logging.info("Feature engineering completed and saved in 'data/processed/'.")

    create_tourist_map(df_engineered, save_path='visualizations/tourist_map.html')
    logging.info("Tourist map created and saved.")

    categorical_columns = ['room_type', 'neighbourhood_cleansed', 'host_response_time', 
                           'host_is_superhost', 'host_identity_verified', 'property_type', 
                           'host_acceptance_rate']
    df_encoded = encode_categorical_columns(df_engineered, categorical_columns)

    logging.info('Performing normality tests after feature engineering...')
    columns_to_test = ['price', 'neighbourhood_cleansed', 'room_type', 'availability_365']
    test_normality(df_encoded, columns=columns_to_test, save_path='results/normality_tests', file_suffix='after_feature_engineering')
    logging.info("Normality tests after feature engineering saved to 'results/normality_tests'.")

    correlations_folder = 'visualizations/correlations'
    if not os.path.exists(correlations_folder):
        os.makedirs(correlations_folder)
        logging.info(f"Created folder: {correlations_folder}")
    else:
        logging.info(f"Folder already exists: {correlations_folder}")

    corr_matrix = calculate_correlation_matrix(df_encoded, method='spearman')

    plot_correlation_matrix(corr_matrix, save_path='visualizations/correlations/correlation_matrix_spearman.png')
    logging.info("Correlation matrix saved to 'visualizations/correlations'.")

    correlated_pairs_df = find_significant_correlations(corr_matrix)
    correlated_pairs_df.to_csv('results/significant_correlations.csv', index=False)
    logging.info("Correlated variables saved in 'results/significant_correlations.csv'.")

if __name__ == '__main__':
    main()
