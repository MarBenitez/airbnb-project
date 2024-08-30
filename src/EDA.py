import pandas as pd

def analyze_missing_values(df: pd.DataFrame):
    """
    Analyze missing values in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
        pd.Series: A series containing the count of missing values per column, sorted by the count.
    """
    missing_values = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
    return missing_values

def basic_info(df: pd.DataFrame):
    """
    Display basic information about the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to display information about.
    """
    print('\nDataframe head:')
    print(df.head(2))
    print('\nDataframe columns:')
    print(df.columns)
    print('\nMissing values:')
    print(analyze_missing_values(df))
    print('\nDataframe info:')
    print(df.dtypes)
