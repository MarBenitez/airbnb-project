# Airbnb Rio de Janeiro Analysis

![Status](https://img.shields.io/badge/status-under_construction-yellow)

This repository contains a refactored and modularized version of the original project developed in a Jupyter Notebook. The objective is to conduct a comprehensive analysis of the Airbnb dataset for Rio de Janeiro and to develop an interactive application that will assist travelers in finding their ideal accommodation.

## 🚧 Project Status

**This project is currently under active development.** Some features may not be fully implemented, and the system might undergo significant changes as improvements are made. Please feel free to explore, provide feedback, or contribute!

 ![Banner](https://images.adsttc.com/media/images/6449/6c61/2def/3401/7a85/1139/slideshow/a-historia-do-calcadao-de-copacabana_3.jpg?1682533486)

## Objective

The main objective of this project is to perform an in-depth analysis of the Airbnb dataset for Rio de Janeiro. The project includes the following key tasks:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Variable Correlation Analysis

In addition, the project aims to develop an interactive application that will be useful for travelers to Rio de Janeiro in finding their ideal accommodation.

## Data Source

The data used in this project has been extracted from [InsideAirbnb](https://insideairbnb.com/rio-de-janeiro/), a platform that provides detailed and publicly available datasets on Airbnb listings in cities around the world. InsideAirbnb was created to enable communities to understand the impact of short-term rentals on housing and neighborhoods. It offers a variety of datasets that include information on listing characteristics, host details, availability, and pricing, which can be used for various analyses including market trends, housing affordability, and neighborhood impacts.

The dataset for Rio de Janeiro includes comprehensive information about the available Airbnb listings in the city, providing a valuable resource for understanding the dynamics of short-term rentals in this popular tourist destination.

## Project Structure

The project follows a well-structured modular design, split into different directories and files based on the function they perform.

- **`data/`**: Contains all data-related files and directories.
  - **`raw/`**: Contains the raw, unprocessed data files.
  - **`intermediate/`**: Contains data files that are in intermediate stages of processing.
  - **`processed/`**: Contains the processed data files after cleaning and feature engineering steps.
  
- **`results/`**: Contains the output files of analysis like normality test results and correlations.
  - **`normality_tests/`**: Contains JSON files with the results of normality tests conducted at various stages.
  - **`significant_correlations.csv`**: Stores the significant correlation pairs found in the correlation matrix analysis.

- **`visualizations/`**: Contains visual outputs such as histograms, boxplots, and correlation matrices.
  - **`cleaned_data/`**: Visualizations after data cleaning.
  - **`encoded/`**: Visualizations after encoding categorical variables.
  - **`engineered/`**: Visualizations after feature engineering.
  - **`preprocessed/`**: Visualizations after the initial data preprocessing.
  - **`correlations/`**: Contains the correlation matrix visualizations.
  - **`tourist_map.html`**: Interactive map showing tourist spots and Airbnb properties.
  
- **`notebooks/`**: Contains the Jupyter notebook for exploration (`airbnb_rio.ipynb`).
  
- **`src/`**: Contains the core Python modules.
  - **`correlations.py`**: Functions for calculating correlations and performing normality tests.
  - **`data_cleaning.py`**: Functions to handle missing values, outliers, and data cleaning.
  - **`data_loading.py`**: Functions to load the dataset.
  - **`data_preprocessing.py`**: Functions to preprocess and transform data.
  - **`EDA.py`**: Functions to perform Exploratory Data Analysis.
  - **`feature_engineering.py`**: Functions to create new features from existing ones.
  - **`visualization.py`**: Functions to generate various plots like histograms, boxplots, and maps.

- **`tests/`**: Contains unit tests for validating the core functionality of the project.
  
- **`main.py`**: The main script that orchestrates the execution of the project workflow.
  
- **`requirements.txt`**: Lists the dependencies required for the project.
  
- **`setup.py`**: Configuration file for packaging the project.

## Tasks Breakdown

### 1. Exploratory Data Analysis (EDA)
- Understand the structure and content of the dataset
- Identify key variables and their distributions
- Detect patterns and anomalies
- Visualize the data for better insights

### 2. Data Cleaning
- Handle missing values
- Correct inconsistencies in the data
- Handle outliers and anomalies
- Remove duplicates and irrelevant data

### 3. Feature Engineering
- Create new features to enhance the dataset
- Transform existing variables for better analysis
- Encode categorical variables
- Generate interaction terms if necessary

### 4. Variable Correlation Analysis
- Compute correlation coefficients
- Visualize correlations using heatmaps or scatter plots
- Identify highly correlated variables
- Understand relationships between different features

## Notebooks

To run the analysis:

- **In your local environment:** Simply download and open `airbnb_rio.ipynb` in Jupyter Notebook or Jupyter Lab and follow the code cells. The notebook is thoroughly documented to guide you through the analysis process.

## Interactive Application Development
- Design and implement an [_interactive application_](https://github.com/MarBenitez/airbnb-rio-streamlit-app)
- Ensure a user-friendly interface and navigation
- Integrate the EDA insights to help travelers
- Provide filters and search functionality for accommodations
- Display key metrics and visualizations to aid decision-making

## Tools and Technologies

- **Python** for data analysis and cleaning
- **Pandas** and **NumPy** for data manipulation
- **Matplotlib** and **Seaborn** for data visualization
- **Scikit-learn** for feature engineering
- **Streamlit** for building the interactive application

## Conclusion
By completing this project, we aim to provide valuable insights into the Airbnb market in Rio de Janeiro. The interactive application will serve as a practical tool for travelers to find their ideal accommodation based on data-driven analysis.

## Installation

Clone the repository and install the necessary dependencies with:
```bash
git clone https://github.com/MarBenitez/airbnb-rio-analysis.git
cd airbnb-rio-analysis
pip install -r requirements.txt

