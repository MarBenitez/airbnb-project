import pandas as pd
import geopandas as gpd
from typing import Dict

def load_data(filepaths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load datasets from a dictionary of file paths.

    Parameters:
        filepaths (dict): A dictionary with file names as keys and file paths as values.

    Returns:
        dict: A dictionary with file names as keys and DataFrames as values.
    """
    data = {}
    for name, path in filepaths.items():
        if path.endswith('.geojson'):
            data[name] = gpd.read_file(path)
        else:
            data[name] = pd.read_csv(path)
    return data