import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium

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


# Keep adding functions for more visualizations
