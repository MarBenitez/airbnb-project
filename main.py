from src.data_preprocessing import preprocess_data
from src.EDA import basic_info
from src.visualization import plot_variables, plot_price_by_neighbourhood, plot_boxplot, plot_violin, plot_histogram
from src.data_cleaning import remove_zero_price, handle_outliers, test_normality


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
    for var in variables_to_plot:
        plot_variables(data_cleaned['merged_listing'], var)
    
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
    for var in numeric_vars:
        normality_result = test_normality(df, var)
        print(f'Normality test result for {var}:', normality_result)
        
        # Interpret the result
        alpha = 0.05  # Common threshold for statistical significance
        if normality_result['Shapiro-Wilk']['p-value'] > alpha:
            print(f"The variable '{var}' appears to be normally distributed (fail to reject H0).")
        else:
            print(f"The variable '{var}' does NOT appear to be normally distributed (reject H0).")

    print('Data cleaning and EDA completed successfully.')

if __name__ == '__main__':
    main()