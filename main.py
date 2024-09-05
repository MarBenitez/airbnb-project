import logging
from src.data_preprocessing import preprocess_data
from src.EDA import basic_info
from src.visualization import (plot_price_by_neighbourhood, plot_boxplot, create_tourist_map, 
                               plot_correlation_matrix, plot_correlation_heatmap)
from src.data_cleaning import remove_zero_price, handle_outliers, handle_missing_values
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
    data_preprocessed = preprocess_data(filepaths, save_merged=True)
    logging.info("Data preprocessing completed successfully and files saved in 'data/intermediate/'.")

    df_preprocessed = data_preprocessed['merged_listing']
    df_no_nulls = handle_missing_values(df_preprocessed)
    logging.info("Missing values handled successfully.")

    df_no_nulls = remove_zero_price(df_no_nulls)

    plot_boxplot(df_no_nulls, 'price', title='Boxplot of Price Before Outlier Handling')

    df_cleaned = handle_outliers(df_no_nulls, 'price', method='iqr')

    plot_boxplot(df_cleaned, 'price', title='Boxplot of Price After Outlier Handling')

    logging.info('Normality test results after handling outliers:')
    numeric_vars = df_cleaned.select_dtypes(include='number').columns
    logging.debug(f"Numeric columns: {numeric_vars}")  # Debugging output

    for var in numeric_vars:
        logging.debug(f"Testing normality for column: {var}")  # Debugging step
        try:
            normality_result = test_normality(df_cleaned, [var])
            logging.info(f'Normality test result for {var}: {normality_result}')

            alpha = 0.05  
            if normality_result[var]['is_normal']:
                logging.info(f"The variable '{var}' appears to be normally distributed (fail to reject H0).")
            else:
                logging.info(f"The variable '{var}' does NOT appear to be normally distributed (reject H0).")
        except KeyError as e:
            logging.error(f"Error: {e} - Column {var} not found in DataFrame.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while testing column {var}: {e}")

    df_engineered = perform_feature_engineering(df_cleaned)
    
    df_engineered.to_csv('data/processed/engineered_listing.csv', index=False)
    logging.info("Feature engineering completed and saved in 'data/processed/'.")

    tourist_map = create_tourist_map(df_engineered)
    logging.info("Tourist map created and saved as 'tourist_map.html' in 'visualizations/'.")

    categorical_columns = ['room_type', 'neighbourhood_cleansed', 'host_response_time', 
                           'host_is_superhost', 'host_identity_verified', 'property_type', 
                           'host_acceptance_rate']
    df_encoded = encode_categorical_columns(df_engineered, categorical_columns)

    columns_to_test = ['price', 'neighbourhood_cleansed', 'room_type', 'availability_365']
    normality_results = test_normality(df_encoded, columns_to_test)

    corr_matrix = calculate_correlation_matrix(df_encoded, method='spearman')

    plot_correlation_matrix(corr_matrix)

    plot_correlation_heatmap(corr_matrix)

    correlated_pairs_df = find_significant_correlations(corr_matrix)
    logging.info("Correlated variables with a correlation greater than 0.7 or less than -0.7:")
    logging.info(correlated_pairs_df)

if __name__ == '__main__':
    main()
