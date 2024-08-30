from src.data_preprocessing import preprocess_data
from src.EDA import analyze_missing_values, basic_info
from src.visualization import plot_variables, plot_price_by_neighbourhood


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
    
    # Save the cleaned data
    data_cleaned['merged_listing'].to_csv('data/processed/merged_listing_preproc.csv', index=False)
    
    print("Data preprocessing completed successfully and files saved in 'data/processed/'.")

    # Perform EDA
    analyze_missing_values(data_cleaned['merged_listing'])
    basic_info(data_cleaned['merged_listing'])

    # Generate Visualizations
    plot_variables(data_cleaned['merged_listing'], 'price')
    plot_price_by_neighbourhood(data_cleaned['merged_listing'])

if __name__ == '__main__':
    main()