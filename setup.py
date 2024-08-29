from setuptools import setup, find_packages

setup(
    name='airbnb-project',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy',
        'statsmodels',
        'geopy',
        'plotly',
        'folium',
        'geopandas',
        # Keep adding packages here
    ],
    description='A comprehensive analysis of Airbnb data for Rio de Janeiro',
    author='Mar Benitez',
    author_email='mar27benitez@gmail.com',
    url='https://github.com/MarBenitez/airbnb-project',
)
