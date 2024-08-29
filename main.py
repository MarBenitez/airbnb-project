from src.data_preprocessing import preprocess_data

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
    data_cleaned['merged_listing'].to_csv('data/processed/merged_listing_cleaned.csv', index=False)
    
    print("Data preprocessing completed successfully and files saved in 'data/processed/'.")

if __name__ == '__main__':
    main()
