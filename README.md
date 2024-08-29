# Airbnb Rio de Janeiro Analysis

![Status](https://img.shields.io/badge/status-under_construction-yellow)

This repository contains a refactored and modularized version of the original project developed in a Jupyter Notebook. The objective is to conduct a comprehensive analysis of the Airbnb dataset for Rio de Janeiro, and to develop an interactive application that will assist travelers in finding their ideal accommodation.

**This project is currently under active development.** Some features may not be fully implemented, and the system might undergo significant changes as improvements are made. Please feel free to explore, provide feedback, or contribute!

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

- `data/`: Contains the raw data.
- `notebooks/`: Contains the original notebooks.
- `src/`: Contains Python modules with functions for data processing, feature engineering, modeling, and visualization.
- `tests/`: Contains unit tests for the modules.

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
- **Google Colab:** Access the project on Google Drive, follow the link to the folder, and open `airbnb_rio_colab.ipynb` in Colab.

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
- **Streamlit** or **Dash** for building the interactive application

## Requirements

Install the necessary packages with:
```bash
pip install -r requirements.txt
```

## Conclusion

By completing this project, we aim to provide valuable insights into the Airbnb market in Rio de Janeiro. The interactive application will serve as a practical tool for travelers to find their ideal accommodation based on data-driven analysis.

---

This refactoring process involved modularizing the original notebook into a well-structured repository, making it easier to maintain, collaborate, and extend in the future.
