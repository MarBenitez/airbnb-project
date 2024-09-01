import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from geopy.distance import geodesic
from folium.plugins import FastMarkerCluster

def map_categories_to_colors(series):
    """
    Map categories to colors.
    
    Parameters:
        series: pandas Series
    
    Returns:
        category_colors, category_color_map
    """
    unique_categories = series.unique()
    colors = px.colors.qualitative.Plotly
    category_color_map = {category: colors[i % len(colors)] for i, category in enumerate(unique_categories)}
    category_colors = series.map(category_color_map)
    return category_colors, category_color_map

def plot_variables(df, var):
    """
    Plot the distribution of a variable by different categories.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        var (str): The variable name.
    """
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            f'{var} distribution',
            f'{var} boxplot',
            f'{var} vs price',
            f'{var} vs room type by room type',
            f'{var} vs neighbourhood by neighbourhood',
            f'{var} vs superhost status by superhost status',
            f'{var} vs review_scores_rating by room_type'
        ),
        horizontal_spacing=0.15, vertical_spacing=0.1
    )

    # Add plots
    fig.add_trace(go.Histogram(x=df[var]), row=1, col=1)
    fig.add_trace(go.Box(x=df[var]), row=1, col=2)
    fig.add_trace(go.Scatter(x=df[var], y=df['price'], mode='markers'), row=2, col=1)
    
    # Add more plots
    room_type_colors, _ = map_categories_to_colors(df['room_type'])
    fig.add_trace(go.Scatter(x=df[var], y=df['price'], mode='markers', marker=dict(color=room_type_colors)), row=2, col=2)
    
    neighbourhood_colors, _ = map_categories_to_colors(df['neighbourhood_cleansed'])
    fig.add_trace(go.Scatter(x=df[var], y=df['price'], mode='markers', marker=dict(color=neighbourhood_colors)), row=3, col=1)

    superhost_colors, _ = map_categories_to_colors(df['host_is_superhost'])
    fig.add_trace(go.Scatter(x=df[var], y=df['price'], mode='markers', marker=dict(color=superhost_colors)), row=3, col=2)

    fig.add_trace(go.Scatter(x=df[var], y=df['review_scores_rating'], mode='markers', marker=dict(color=room_type_colors)), row=4, col=1)

    # Layout update
    fig.update_layout(height=1200, width=1000, title_text=f'Analysis of {var}', showlegend=False)
    fig.show()

def plot_price_by_neighbourhood(df):
    """
    Plot the average price by neighbourhood.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
    """
    price_neighbourhood = df.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False)
    fig = px.bar(price_neighbourhood, x=price_neighbourhood.index, y=price_neighbourhood.values, title='Price by Neighbourhood',
                 labels={'x': 'Neighbourhood', 'y': 'Price'})
    fig.show()

def plot_boxplot(df, column, title="Boxplot"):
    """
    Plot a boxplot for a given column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name to plot.
        title (str): The title of the plot.
    """
    fig = px.box(df, y=column, points='all', title=title)
    fig.show()

def plot_violin(df, x_column, y_column=None, title="Violin Plot"):
    """
    Plot a violin plot for the given columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x_column (str): The x-axis column name.
        y_column (str): The y-axis column name (optional).
        title (str): The title of the plot.
    """
    if y_column:
        fig = px.violin(df, x=x_column, y=y_column, title=title)
    else:
        fig = px.violin(df, x=x_column, title=title)
    fig.show()

def plot_histogram(df, column, title="Histogram"):
    """
    Plot a histogram for a given column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name to plot.
        title (str): The title of the plot.
    """
    fig = px.histogram(df, x=column, title=title)
    fig.show()



def create_tourist_map(df, save_path='visualizations/tourist_map2.html'):
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
    map1 = folium.Map(location=map_center, zoom_start=11.5)

    # Add the tourist spots with distinctive icons
    for place in tourist_locations:
        folium.Marker(
            location=place["location"],
            popup=place["name"],
            icon=folium.Icon(color='blue', prefix='fa', icon=place["icon"])
        ).add_to(map1)

    # Add clustered property markers to the map
    lats = df['latitude'].tolist()
    lons = df['longitude'].tolist()
    locations = list(zip(lats, lons))
    FastMarkerCluster(data=locations).add_to(map1)

    # Save the map as an HTML file
    map1.save(save_path)
    print(f"Map saved as {save_path}")


# Keep adding functions for more visualizations
