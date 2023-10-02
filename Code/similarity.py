import numpy as np
import pandas as pd

def search_dates(historical: pd.DataFrame, datapoint: pd.DataFrame, num_dates:int = 1, threshold:float = .2):
    """
    Function to find close historical datapoints
    
    Parameters
    ----------
    historical : pd.DataFrame
        historical data for reference (only the data of interest --- controllable variables) 
    datapoint : pd.DataFrame
        new/incoming optimized data (only the data of interest --- controllable variables)
    num_dates : int
        number of similar historical records to return
    threshold : float, default=.1
        similarity percent to match (this may not be used ...)

    Returns
    -------
    _ : np.ndarray
        numpy array, shape (num_dates, len(controllable_vars)) holding the 4 closest historical controllable settings to the calculated optimal values
    """
    # NOTE: historical should only hold the controllable variables
    historical_copy = historical.copy()

    # NOTE: use mape for distance metric since this will allow for easy identification of "close points" that satisfy the threshold
    # distances = np.sqrt(np.square((historical_copy - datapoint) / datapoint ).sum(axis=1))
    distances =  ( (historical_copy - datapoint) / historical_copy ).abs().mean(axis=1)
    historical_copy['Distance'] = distances

    # filter to distances that are closer than 20% to the actual datapoint
    historical_copy = historical_copy.loc[historical_copy['Distance'] <= threshold, :].copy()


    # sort by distance, drop distance and return 
    return historical_copy.sort_values(by=['Distance']).drop(labels='Distance', axis=1).iloc[0:num_dates, :]