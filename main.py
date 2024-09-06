import logging
from src.data_loading import load_data
from src.data_preprocessing import load_preprocess_data
from src.EDA import basic_info
from src.visualization import (
    plot_price_by_neighbourhood, plot_boxplot, create_tourist_map, 
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

    # 7. Perform normality tests on relevant columns
    logging.info('Normality test results after handling outliers:')
    numeric_vars = df_cleaned.select_dtypes(include='number').columns

    for var in numeric_vars:
        try:
            normality_result = test_normality(df_cleaned, [var], save_path=f'results/normality_test_{var}.json')
            logging.info(f'Normality test result for {var} saved to file.')
        except KeyError as e:
            logging.error(f"Error: {e} - Column {var} not found in DataFrame.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while testing column {var}: {e}")
    
    # 8. Feature Engineering
    df_engineered = perform_feature_engineering(df_cleaned)
    df_engineered.to_csv('data/processed/engineered_listing.csv', index=False)
    logging.info("Feature engineering completed and saved in 'data/processed/'.")

    # 9. Create tourist map visualization
    tourist_map = create_tourist_map(df_engineered, save_path='visualizations/tourist_map.html')
    logging.info("Tourist map created and saved.")

    # 10. Encode categorical columns
    categorical_columns = ['room_type', 'neighbourhood_cleansed', 'host_response_time', 
                           'host_is_superhost', 'host_identity_verified', 'property_type', 
                           'host_acceptance_rate']
    df_encoded = encode_categorical_columns(df_engineered, categorical_columns)

    # 11. Perform normality tests again if necessary (post-feature engineering)
    columns_to_test = ['price', 'neighbourhood_cleansed', 'room_type', 'availability_365']
    normality_results = test_normality(df_encoded, columns_to_test, save_path='results/post_feature_normality.json')

    # 12. Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(df_encoded, method='spearman')

    # 13. Plot correlation matrix and heatmap
    plot_correlation_matrix(corr_matrix, save_path='visualizations/correlation_matrix.png')

    # 14. Find and log significant correlations
    correlated_pairs_df = find_significant_correlations(corr_matrix)
    correlated_pairs_df.to_csv('results/significant_correlations.csv', index=False)
    logging.info("Correlated variables saved in 'results/significant_correlations.csv'.")

if __name__ == '__main__':
    main()
