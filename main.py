from src.data_preprocessing import preprocess_data
from src.EDA import basic_info
from src.visualization import plot_variables, plot_price_by_neighbourhood, plot_boxplot, plot_violin, plot_histogram, create_tourist_map
from src.data_cleaning import remove_zero_price, handle_outliers
from src.feature_engineering import perform_feature_engineering
from src.correlations import (
    encode_categorical_columns, 
    test_normality, 
    calculate_correlation_matrix, 
    plot_correlation_matrix, 
    plot_correlation_heatmap, 
    find_significant_correlations
)
from sklearn.preprocessing import LabelEncoder


def main():
    filepaths = {
        'listing': 'data/raw/listings.csv',
        'listing_details': 'data/raw/listings.csv.gz',
        'reviews': 'data/raw/reviews.csv',
        'reviews_details': 'data/raw/reviews.csv.gz',
        'neighbourhoods': 'data/raw/neighbourhoods.csv',
        'neighbourhoods_geo': 'data/raw/neighbourhoods.geojson',
        'calendar': 'data/raw/calendar.csv.gz'
    }
    
    # Preprocess the data and save the merged listing if needed
    data_cleaned = preprocess_data(filepaths, save_merged=True)
    
    print("Data preprocessing completed successfully and files saved in 'data/intermediate/'.")

    # Perform EDA
    basic_info(data_cleaned['merged_listing'])

    # Define the variables to plot
    variables_to_plot = [
        'price', 'accommodates', 'beds', 'room_type',
        'review_scores_rating', 'minimum_nights', 'maximum_nights', 'host_is_superhost'
    ]
    
    # Generate visualizations for each variable in the list
    #for var in variables_to_plot:
        #plot_variables(data_cleaned['merged_listing'], var)
    
    # Plot price by neighbourhood
    plot_price_by_neighbourhood(data_cleaned['merged_listing'])

    # Cleaned data
    ## Remove rows where price is zero
    df = data_cleaned['merged_listing']
    df = remove_zero_price(df)

    ## Plot before handling outliers
    plot_boxplot(df, 'price', title='Boxplot of Price Before Outlier Handling')

    ## Handle 'price' outliers using IQR
    df = handle_outliers(df, 'price', method='iqr')

    ## Plot after handling outliers
    plot_boxplot(df, 'price', title='Boxplot of Price After Outlier Handling')

    # Test normality for each variable after handling outliers
    print('Normality test results after handling outliers:')

    numeric_vars = df.select_dtypes(include='number').columns
    print(f"Numeric columns: {numeric_vars}")  # Debugging output

    for var in numeric_vars:
        print(f"Testing normality for column: {var}")  # Debugging step
        try:
            normality_result = test_normality(df, [var])
            print(f'Normality test result for {var}:', normality_result)

            # Interpret the result
            alpha = 0.05  # Common threshold for statistical significance
            if normality_result[var]['is_normal']:
                print(f"The variable '{var}' appears to be normally distributed (fail to reject H0).")
            else:
                print(f"The variable '{var}' does NOT appear to be normally distributed (reject H0).")
        except KeyError as e:
            print(f"Error: {e} - Column {var} not found in DataFrame.")
        except Exception as e:
            print(f"An unexpected error occurred while testing column {var}: {e}")


    # Feature Engineering
    df_engineered = perform_feature_engineering(df)
    
    # Save the engineered data
    df_engineered.to_csv('data/processed/engineered_listing.csv', index=False)
    
    print("Feature engineering completed and saved in 'data/processed/'.")

    # Crear y guardar el mapa de los lugares turísticos
    tourist_map = create_tourist_map(df_engineered)

    print("Tourist map created and saved as 'tourist_map.html' in 'visualizations/'.")

    # Encode categorical variables
    categorical_columns = ['room_type', 'neighbourhood_cleansed', 'host_response_time', 
                           'host_is_superhost', 'host_identity_verified', 'property_type', 
                           'host_acceptance_rate']
    df_encoded = encode_categorical_columns(df, categorical_columns)

    # Test for normality
    columns_to_test = ['price', 'neighbourhood_cleansed', 'room_type', 'availability_365']
    normality_results = test_normality(df_encoded, columns_to_test)

    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(df_encoded, method='spearman')

    # Plot the correlation matrix
    plot_correlation_matrix(corr_matrix)

    # Plot using Plotly
    plot_correlation_heatmap(corr_matrix)

    # Find and display significant correlations
    correlated_pairs_df = find_significant_correlations(corr_matrix)
    print("Variables correlacionadas con una correlación mayor a 0.7 o menor a -0.7:")
    print(correlated_pairs_df)

if __name__ == '__main__':
    main()