import os
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import FastMarkerCluster
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_price_by_neighbourhood(df, save_path=None):
    """
    Plot the average price by neighbourhood.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        save_path (str): Path to save the plot.
    """
    if 'neighbourhood_cleansed' not in df.columns:
        raise KeyError("The column 'neighbourhood_cleansed' is missing from the DataFrame.")
    
    price_neighbourhood = df.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=price_neighbourhood.index, y=price_neighbourhood.values)
    plt.title('Price by Neighbourhood')
    plt.xticks(rotation=90)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Plot saved successfully to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_boxplot(df, column, title="Boxplot", save_path=None):
    """
    Plot a boxplot for a given column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name to plot.
        title (str): The title of the plot.
        save_path (str): Path to save the plot. If None, it will display the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column])
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Boxplot saved successfully to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_histogram(df, column, title="Histogram", save_path=None):
    """
    Plot a histogram for a given column and optionally save it to a file.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name to plot.
        title (str): The title of the plot.
        save_path (str): Path to save the plot. If None, it will display the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Histogram saved successfully to {save_path}")
    else:
        plt.show()

    plt.close()


def create_tourist_map(df, save_path='visualizations/tourist_map.html'):
    """
    Create an interactive map showing all tourist locations with clustered property markers.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the property data.
        save_path (str): Path where the map HTML file will be saved.

    Returns:
        None: The function saves the map as an HTML file.
    """
    # Coordinates of the tourist spots with distinctive icons
    tourist_locations = [
        {"name": "Cristo Redentor", "location": [-22.9519, -43.2105], "icon": 'fa-monument'},
        {"name": "Pão de Açúcar", "location": [-22.9486, -43.1553], "icon": 'fa-mountain'},
        {"name": "Praia de Copacabana", "location": [-22.9711, -43.1822], "icon": 'fa-umbrella-beach'},
        {"name": "Maracanã", "location": [-22.9839, -43.2045], "icon": 'fa-futbol'},
        {"name": "Jardim Botânico", "location": [-22.9674, -43.2292], "icon": 'fa-leaf'}
    ]

    # Create a map centered on Rio de Janeiro
    map_center = [-22.90642, -43.18223]
    tourist_map = folium.Map(location=map_center, zoom_start=11.5)

    # Add the tourist spots with distinctive icons
    for place in tourist_locations:
        folium.Marker(
            location=place["location"],
            popup=place["name"],
            icon=folium.Icon(color='blue', prefix='fa', icon=place["icon"])
        ).add_to(tourist_map)

    # Add clustered property markers to the map
    lats = df['latitude'].tolist()
    lons = df['longitude'].tolist()
    locations = list(zip(lats, lons))
    FastMarkerCluster(data=locations).add_to(tourist_map)

    # Save the map as an HTML file
    tourist_map.save(save_path)
    logging.info(f"Map saved as {save_path}")


def plot_correlation_matrix(corr_matrix, save_path=None):
    """
    Plot the correlation matrix using a heatmap.

    Parameters:
        corr_matrix: Correlation matrix to plot.
        save_path: Path to save the plot. If None, it will display the plot.
    """
    plt.figure(figsize=(10, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt=".2f")

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Correlation matrix saved successfully to {save_path}")
    else:
        plt.show()

    plt.close()
